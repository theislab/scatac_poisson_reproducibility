from typing import Dict, Iterable, Optional, Callable

import numpy as np
import torch
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl
import torch.nn as nn


from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder, FCLayers, DecoderSCVI


class DecoderPEAKVI(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions into ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        use_region_factor: bool = True
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.use_region_factor = use_region_factor
        if use_region_factor:
            self.region_factor = torch.nn.Parameter(torch.zeros(n_output))
            
        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        if self.use_region_factor:
            px_scale = px_scale + self.region_factor
        px_scale = torch.softmax(px_scale, dim = -1)
        
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = None
        return px_scale, px_r, px_rate, px_dropout


class PEAKVAE(BaseModuleClass):
    """
    Variational auto-encoder model for ATAC-seq data.

    This is an implementation of the peakVI model descibed in.

    Parameters
    ----------
    n_input_regions
        Number of input regions.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer. If `None`, defaults to square root
        of number of regions.
    n_latent
        Dimensionality of the latent space. If `None`, defaults to square root
        of `n_hidden`.
    n_layers_encoder
        Number of hidden layers used for encoder NN.
    n_layers_decoder
        Number of hidden layers used for decoder NN.
    dropout_rate
        Dropout rate for neural networks
    model_depth
        Model library size factors or not.
    region_factors
        Include region-specific factors in the model
    use_batch_norm
        One of the following

        * ``'encoder'`` - use batch normalization in the encoder only
        * ``'decoder'`` - use batch normalization in the decoder only
        * ``'none'`` - do not use batch normalization (default)
        * ``'both'`` - use batch normalization in both the encoder and decoder
    use_layer_norm
        One of the following

        * ``'encoder'`` - use layer normalization in the encoder only
        * ``'decoder'`` - use layer normalization in the decoder only
        * ``'none'`` - do not use layer normalization
        * ``'both'`` - use layer normalization in both the encoder and decoder (default)
    latent_distribution
        which latent distribution to use, options are

        * ``'normal'`` - Normal distribution (default)
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers of the decoder. If False (default),
        covairates will only be included in the input layer.

    """

    def __init__(
        self,
        n_input_regions: int,
        n_batch: int = 0,
        n_hidden: Optional[int] = None,
        n_latent: Optional[int] = None,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        model_depth: bool = True,
        region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: str = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        log_variational: bool = True,
    ):
        super().__init__()

        self.n_input_regions = n_input_regions
        self.n_hidden = (
            int(np.sqrt(self.n_input_regions)) if n_hidden is None else n_hidden
        )
        self.n_latent = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.model_depth = model_depth
        self.dropout_rate = dropout_rate
        self.latent_distribution = latent_distribution
        self.use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        self.use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        self.use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        self.use_layer_norm_decoder = use_layer_norm in ("decoder", "both")
        self.deeply_inject_covariates = deeply_inject_covariates
        self.encode_covariates = encode_covariates
        self.log_variational = log_variational
        
        self.use_observed_lib_size = use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_means is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )
            
        cat_list = (
            [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else []
        )

        n_input_encoder = self.n_input_regions + n_continuous_cov * encode_covariates
        encoder_cat_list = cat_list if encode_covariates else None
        self.z_encoder = Encoder(
            n_input=n_input_encoder,
            n_layers=self.n_layers_encoder,
            n_output=self.n_latent,
            n_hidden=self.n_hidden,
            n_cat_list=encoder_cat_list,
            dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution=self.latent_distribution,
            var_eps=0,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
        )

        n_input_decoder = self.n_latent + n_continuous_cov
        print(n_input_decoder)
        self.region_factors = region_factors
        self.z_decoder = DecoderPEAKVI(
            n_input_decoder,
            self.n_input_regions,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            n_hidden=self.n_hidden,
            inject_covariates=self.deeply_inject_covariates,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            use_region_factor=self.region_factors
        )

        self.l_encoder = None
        if self.model_depth:
            # Decoder class to avoid variational split
            self.l_encoder = Encoder(
                n_input_encoder,
                1,
                n_layers=1,
                n_cat_list=encoder_cat_list,
                n_hidden=self.n_hidden,
                dropout_rate=self.dropout_rate,
                inject_covariates=self.deeply_inject_covariates,
                use_batch_norm=self.use_batch_norm_encoder,
                use_layer_norm=self.use_layer_norm_encoder,
                var_activation=var_activation,
            )
            

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
        input_dict = dict(
            x=x,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
        input_dict = {
            "z": z,
            "library": library,
            "batch_index": batch_index,
            "y": y,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }
        return input_dict

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout) -> torch.Tensor:
        reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss
    
    @auto_move_data
    def inference(self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        qz_m, qz_v, z = self.z_encoder(encoder_input, batch_index, *categorical_input)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
                ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
                library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)
        return outputs

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        # TODO: refactor forward function to not rely on y
        decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        px_scale, px_r, px_rate, px_dropout = self.z_decoder(
            decoder_input, library, batch_index, *categorical_input, y
        )
        px_r = None

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )


    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, qz_v.sqrt()), Normal(mean, scale)).sum(dim=1)

        if not self.use_observed_lib_size:
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_divergence_l = kl(
                Normal(ql_m, ql_v.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            ).sum(dim=1)
        else:
            kl_divergence_l = 0.0

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, reconst_loss, kl_local, kl_global)
