"""
Module designed to write .h5 Napari projects
"""
from typing import Callable, Any, Optional
from napari_plugin_engine import napari_hook_implementation
import numpy as np
import sparse
import h5py


@napari_hook_implementation(specname='napari_get_writer')
def project_to_h5(path: str) -> Callable or None:
    """Returns a h5 Napari project writer if the path file format is h5.

    Parameters
    ----------
    path: str
        Napari h5 project file

    Returns
    -------
    Callable or None
        Napari h5 project file writer if the path file extension is correct
    """
    if isinstance(path, str) and path.endswith('.h5'):
        return write_layers_h5
    else:
        return None


@napari_hook_implementation
def napari_write_image(path: str, data: Any, meta: dict) -> Optional[str]:
    return layer_writer(path, data, meta, layer_type='image')


@napari_hook_implementation
def napari_write_labels(path: str, data: Any, meta: dict) -> Optional[str]:
    return layer_writer(path, data, meta, layer_type='labels', sparse=True)


@napari_hook_implementation
def napari_write_points(path: str, data: Any, meta: dict) -> Optional[str]:
    return layer_writer(path, data, meta, layer_type='points')


def write_layers_h5(path, layer_data) -> str:
    """Returns a list of LayerData tuples from the
    project file required by Napari.

    Parameters
    ----------
    path: str
        Napari h5 project output file
    layer_data: list[tuple[numpy array, dict, str]]
        List of LayerData tuples produced by Napari IO writer

    Returns
    -------
    str
        Final output file path
    """
    with h5py.File(path, 'w') as hdf:
        # for data, meta, layer_type in layer_data:
        for i, tmp_data in enumerate(layer_data):
            data, meta, layer_type = tmp_data
            meta['pos'] = i  # layer position in napari
            layer_name = meta['name']

            # check if the layer_type group already exists
            if layer_type not in hdf.keys():
                hdf.create_group(layer_type)
            if layer_type == 'labels':
                compressed_shape, compressed_data = compress_layer(data)
                if data.nbytes >= compressed_data.nbytes:
                    meta['shape'], data = compressed_shape, compressed_data
                meta['compressed'] = data.nbytes >= compressed_data.nbytes
            hdf[layer_type].create_dataset(layer_name, data=data)

            # add all metadata as attributes of each layer
            for key, val in process_metadata(meta).items():
                try:
                    hdf[layer_type][layer_name].attrs[key] = val
                except TypeError:
                    pass
    return path


def layer_writer(path: str, data: Any, meta: dict,
                 layer_type: str, sparse: bool = False) -> str or None:
    """Function to write single Napari layers in a h5 project file.

    Parameters
    ----------
    path: str
        Napari h5 project output file.
    data: np.array
        Napari layer data.
    meta: dict
        dictionary of Napari layer metadata.
    layer_type: str
        Napari layer type.
    sparse: bool
        True if Napari layer should be represented in COO list.


    Returns
    -------
    str
        Final output file path
    """
    if isinstance(path, str) and path.endswith('.h5'):
        del meta['data']  # data key which stores the layer data
        layer_name = meta['name']
        meta['pos'] = 0
        if sparse:
            meta['shape'], data = compress_layer(data)
        with h5py.File(path, 'w') as hdf:
            hdf.create_group(layer_type)
            hdf[layer_type].create_dataset(layer_name, data=data)
            for key, val in process_metadata(meta).items():
                try:
                    hdf[layer_type][layer_name].attrs[key] = val
                except TypeError:
                    pass
        return path
    else:
        return None


def compress_layer(layer_array: np.array) -> tuple:
    """Returns a numpy array corresponding to a sparse
    version of the original data array.

    Parameters
    ----------
    layer_array: Numpy array
        Original array version of the layer data

    Returns
    -------
    np.array
        Sparse version (COO list) of the layer data
    """
    initial_shape = tuple(layer_array.shape)
    tmp_coo = sparse.COO(layer_array)
    compressed_array = np.append(tmp_coo.coords, np.array(
        [tmp_coo.data]), axis=0)  # join coords and data in single array
    return initial_shape, compressed_array.astype(np.int16)


def process_metadata(meta_dict: dict) -> dict:
    """Returns a dictionary including the optional layer metadata
    information (stored as a nested dictionary).

    Parameters
    ----------
    meta_dict: dict
        Dictionary of Napari layer attributes

    Returns
    -------
    dict
        dictionary without nested information
        (to store as h5 dataset attributes)
    """
    metadata = meta_dict.pop('metadata')
    for key, val in metadata.items():
        meta_dict['meta_{}'.format(key)] = val
    return meta_dict
