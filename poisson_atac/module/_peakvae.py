from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.distributions import Normal, Poisson, kl_divergence
from torch.nn import PoissonNLLLoss

from scvi import REGISTRY_KEYS as _CONSTANTS
from scvi._compat import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder, FCLayers


class DecoderSCVI(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden=128,
        n_layers=2,
        n_cat_list: Iterable[int] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        deep_inject_covariates: bool = False,
    ):
        super().__init__()

        self.n_output = n_output
        # mean gamma
        self.factor_regressor = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_hidden=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            use_activation=True,
            activation_fn=torch.nn.LeakyReLU,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=0,
            inject_covariates=deep_inject_covariates
        )
        
        self.output = torch.nn.Linear(n_hidden, n_output, bias=False)
        self.region_factor = torch.nn.Parameter(torch.zeros(n_output))
    

    def forward(
        self, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        raw_px_scale = self.output(self.factor_regressor(z, *cat_list))   
        px_scale = torch.softmax(raw_px_scale + self.region_factor, dim=-1)
        px_rate = torch.exp(library) * px_scale
        px_r = None
        px_dropout=None
        return px_scale, px_r, px_rate, px_dropout

class  CountVAE(BaseModuleClass):
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
        log_variational=True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: str = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
    ):
        super().__init__()

        self.n_input_regions = n_input_regions
        self.log_variational = log_variational
        self.n_hidden = (
            int(np.sqrt(self.n_input_regions)) if n_hidden is None else n_hidden
        )
        self.n_latent = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.dropout_rate = dropout_rate
        self.latent_distribution = latent_distribution
        self.use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        self.use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        self.use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        self.use_layer_norm_decoder = use_layer_norm in ("decoder", "both")
        self.deeply_inject_covariates = deeply_inject_covariates
        self.encode_covariates = encode_covariates

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
        self.decoder = DecoderSCVI(
            n_input=n_input_decoder,
            n_output=n_input_regions,
            n_hidden=self.n_hidden,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            deep_inject_covariates=self.deeply_inject_covariates,
        )
        
    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]
        x_chr = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        cont_covs = tensors.get(_CONSTANTS.CONT_COVS_KEY)
        cat_covs = tensors.get(_CONSTANTS.CAT_COVS_KEY)
        input_dict = dict(
            x=x,
            x_chr=x_chr,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        z = inference_outputs["z"]
        qz_m = inference_outputs["qz_m"]
        x = tensors[_CONSTANTS.X_KEY]
        library = torch.log(x.sum(1)).unsqueeze(1)
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        cont_covs = tensors.get(_CONSTANTS.CONT_COVS_KEY)

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        input_dict = {
            "z": z,
            "qz_m": qz_m,
            "library": library,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        x,
        x_chr,
        batch_index,
        cont_covs,
        cat_covs,
        n_samples=1,
    ) -> Dict[str, torch.Tensor]:

        """Helper function used in forward pass."""

        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat([x_, cont_covs], dim=-1)  # changed
        else:
            encoder_input = x_  # changed

        # if encode_covariates is False, cat_list to init encoder is None, so
        # batch_index is not used (or categorical_input, but it's empty)
        qz_m, qz_v, z = self.z_encoder(encoder_input, batch_index, *categorical_input)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        return dict(qz_m=qz_m, qz_v=qz_v, z=z)
    
    @auto_move_data
    def generative(
        self,
        z,
        qz_m,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        use_z_mean=False,
    ):
        """Runs the generative model."""

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        latent = z if not use_z_mean else qz_m
        decoder_input = (
            latent if cont_covs is None else torch.cat([latent, cont_covs], dim=-1)
        )

        px_scale, px_r, px_rate, px_dropout = self.decoder(decoder_input, library, batch_index, *categorical_input)

        return dict(px_rate=px_rate)

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ):
        x_chr = tensors[_CONSTANTS.X_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_rate = generative_outputs["px_rate"]

        kld = kl_divergence(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(0, 1),
        ).sum(dim=1)

        rl = self.get_reconstruction_loss(px_rate, x_chr)

        loss = (rl.sum() + kld * kl_weight).sum()

        return LossRecorder(loss, rl, kld, kl_global=torch.tensor(0.0))

    def get_reconstruction_loss(self, px_rate, x):
        rl = PoissonNLLLoss(reduction='none', log_input=False, full=True)(px_rate, x).sum(dim=-1)
        return rl 

    
class BaselineCountVAE(BaseModuleClass):
    def __init__(
        self,
        n_input_regions: int):

        super().__init__()
        self.n_input_regions = n_input_regions
        self.region_factor = torch.nn.Parameter(torch.zeros(n_input_regions))
        
    def _get_inference_input(self, tensors):
        input_dict = dict()
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        x = tensors[_CONSTANTS.X_KEY]
        library = torch.log(x.sum(1)).unsqueeze(1)
        input_dict = {
            "library": library
        }
        return input_dict

    @auto_move_data
    def inference(
        self
    ) -> Dict[str, torch.Tensor]:
        return dict() 
      
    @auto_move_data
    def generative(
        self,
        library,
        use_z_mean=False
    ):
        """Runs the generative model."""
        px_rate = torch.exp(library) * torch.softmax(self.region_factor, dim=-1) 

        return dict(px_rate=px_rate)

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ):
        x_chr = tensors[_CONSTANTS.X_KEY]
        px_rate = generative_outputs["px_rate"]
        
        rl = self.get_reconstruction_loss(px_rate, x_chr)

        loss = (rl.sum()).sum()

        return LossRecorder(loss, rl, kld=torch.tensor(0.0), kl_global=torch.tensor(0.0))

    def get_reconstruction_loss(self, px_rate, x):
        # l is lambda of poisson
        # x is target counts
        #rl = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        rl = PoissonNLLLoss(reduction='none', log_input=False, full=True)(px_rate, x).sum(dim=-1)
        return rl
