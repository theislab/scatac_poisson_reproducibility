from anndata import AnnData
import numpy as np
from typing import Optional


def reads_to_fragments(
    adata: AnnData,
    layer: Optional[str] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
):
    """
    Function to convert read counts to appoximate fragment counts

    Parameters
    ----------
    adata
        AnnData object that contains read counts.
    layer
        Layer that the read counts are stored in.
    key_added
        Name of layer where the fragment counts will be stored.
    copy
        Whether to modify copied input object.
    """
    if copy:
        adata = adata.copy()

    if layer:
        data = np.ceil(adata.layers[layer].data / 2)
    else:
        data = np.ceil(adata.X.data / 2)

    if key_added:
        adata.layers[key_added] = adata.X.copy()
        adata.layers[key_added].data = data
    elif layer and key_added is None:
        adata.layers[layer].data = data
    elif layer is None and key_added is None:
        adata.X.data = data
    if copy:
        return adata
