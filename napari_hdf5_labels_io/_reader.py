"""
Module designed to read .h5 Napari projects
"""
from typing import Callable
from napari_plugin_engine import napari_hook_implementation
import numpy as np
import sparse
import h5py
import warnings


@napari_hook_implementation(specname='napari_get_reader')
def h5_to_napari(path: str) -> Callable or None:
    """Returns a h5 Napari project reader if the path file format is h5.

    Parameters
    ----------
    path: str
        Napari h5 project file

    Returns
    -------
    Callable or None
        Napari h5 project file reader if the path file extension is correct
    """
    if isinstance(path, str) and path.endswith('.h5'):
        return read_layer_h5
    else:
        return None


def read_layer_h5(path: str):
    """Returns a list of LayerData tuples from the project
    file required by Napari.

    Parameters
    ----------
    path: str
        Napari h5 project file

    Returns
    -------
    list[tuple[numpy array, dict, str]]
        List of LayerData tuples required by Napari IO reader
    """
    # use of list.pop() to delete the temporal attributes
    output_dict = {}
    with h5py.File(path, 'r') as hdf:
        for layer_type in hdf:  # iterate over layer types
            for ith_layer in hdf[layer_type]:  # iterate over each layer
                tmp_layer = hdf[layer_type][ith_layer]
                layer_data = np.array(tmp_layer)
                layer_meta = dict(tmp_layer.attrs)
                if layer_type == 'labels':
                    original_shape = layer_meta.pop('shape')
                    try:
                        is_sparse = layer_meta.pop('is_sparse')
                    except KeyError:
                        warnings.warn(
                            "This file has an older plugin version.")
                        is_sparse = True
                    if is_sparse:
                        layer_data = reconstruct_layer(tmp_layer,
                                                       tuple(original_shape))
                    else:
                        layer_data = tmp_layer[:] # read data from zarr array
                layer_position = layer_meta.pop('pos')
                output_dict[layer_position] = (layer_data,
                                               reconstruct_metadata(
                                                   layer_meta),
                                               layer_type)
    layer_data_list = [output_dict[key] for key in sorted(output_dict.keys())]
    return layer_data_list


def reconstruct_layer(layer_array: np.array, shape: tuple) -> np.array:
    """Returns a numpy array corresponding to the data reconstruction from
    a sparse version (COO list) of it.

    Parameters
    ----------
    layer_array: Numpy array
        Sparse array version (COO list) of the layer data
    shape: tuple
        shape of the original layer data array

    Returns
    -------
    np.array
        Full version of the layer data
    """
    layer_array = np.array(layer_array, dtype=np.uint16)
    coords = layer_array[:-1]
    values = layer_array[-1]
    tmp_sparse = sparse.COO(coords=coords, data=values, shape=shape)
    return tmp_sparse.todense()


def reconstruct_metadata(meta_dict: dict) -> dict:
    """Returns a dictionary following the Napari specifications for
    layer meta data.

    Parameters
    ----------
    meta_dict: dict
        Dictionary of h5 dataset attributes (taken from a Napari layer)

    Returns
    -------
    dict
        dictionary with the Napari meta data key's distribution
    """
    metadata = {key[5:]: val for key,
                val in meta_dict.items() if key.startswith('meta_')}
    meta_dict = {key: val for key,
                 val in meta_dict.items() if not key.startswith('meta_')}
    meta_dict['metadata'] = metadata
    return meta_dict
