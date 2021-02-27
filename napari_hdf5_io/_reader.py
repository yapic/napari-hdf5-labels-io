from napari_plugin_engine import napari_hook_implementation
import numpy as np
import sparse
import h5py


@napari_hook_implementation(specname='napari_get_reader')
def h5_to_napari(path):
    if isinstance(path, str) and path.endswith('.h5'):
        return read_layer_h5
    else:
        return None


def read_layer_h5(path):
    output_list = []
    with h5py.File(path, 'r') as hdf:
        for layer_type in hdf:  # iterate over layer types
            for ith_layer in hdf[layer_type]:  # iterate over each layer
                tmp_layer = hdf[layer_type][ith_layer]
                layer_data = np.array(tmp_layer)
                layer_meta = {key: val for key, val in tmp_layer.attrs.items()}
                if layer_type == 'labels':
                    original_shape = layer_meta.pop(
                        'shape')  # delete tmp shape attribute
                    layer_data = reconstruct_layer(layer_data, original_shape)
                output_list.append((layer_data, layer_meta, layer_type))
    return output_list


def reconstruct_layer(layer_array, shape):
    coords = layer_array[:-1]
    values = layer_array[-1]
    tmp_sparse = sparse.COO(coords=coords, data=values, shape=shape)
    return tmp_sparse.todense()
