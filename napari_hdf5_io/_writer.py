from napari_plugin_engine import napari_hook_implementation
import numpy as np
import sparse
import h5py


@napari_hook_implementation(specname='napari_get_writer')
def project_to_h5(path, layer_types):
    if isinstance(path, str) and path.endswith('.h5'):
        return write_layers_h5
    else:
        return None


def write_layers_h5(path, layer_data):
    with h5py.File(path, 'w') as hdf:
        # for data, meta, layer_type in layer_data:
        for i, tmp_data in enumerate(layer_data):
            data, meta, layer_type = tmp_data
            meta['pos'] = i # layer position in napari
            layer_name = meta['name']
            if layer_type not in hdf.keys():  # check if the layer_type group already exists
                hdf.create_group(layer_type)
            if layer_type == 'labels':
                # make compression in the same data variable
                meta['shape'], data = compress_layer(data)
            hdf[layer_type].create_dataset(layer_name, data=data)
            for key, val in process_metadata(meta).items():  # to add all metadata as attributes of each layer
                try:
                    hdf[layer_type][layer_name].attrs[key] = val
                except TypeError:
                    pass
    return path


def compress_layer(layer_array):
    initial_shape = tuple(layer_array.shape)
    tmp_coo = sparse.COO(layer_array)
    compressed_array = np.append(tmp_coo.coords, np.array(
        [tmp_coo.data]), axis=0)  # join coords and data in single array
    return initial_shape, compressed_array.astype(np.int32)


def process_metadata(meta_dict):
    metadata = meta_dict.pop('metadata')
    for key, val in metadata.items():
        meta_dict['meta_{}'.format(key)] = val
    return meta_dict
