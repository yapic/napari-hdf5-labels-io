from napari_hdf5_labels_io._writer import write_layers_h5, layer_writer, compress_layer, process_metadata
import numpy as np
import os

test_meta = {'key1': 0, 'metadata': {'a': 0, 'b': 1}, 'key3': 2}
test_layer_data = np.zeros((3,3))
test_layer_data[1, 1] = 1

def test_process_metadata():
    out_dict = {'key1': 0, 'meta_a': 0, 'meta_b': 1, 'key3': 2}
    assert process_metadata(test_meta) == out_dict


def test_compress_layer():
    out_data = np.array([[1], [1], [1]])
    out_shape = (3, 3)
    shape, data = compress_layer(test_layer_data)
    assert out_shape == shape
    assert all(data == out_data)

def test_layer_writer(tmp_path):
    test_single_data = {'key1': 0, 'metadata': {'a': 0, 'b': 1}, 'data': [1, 2, 3], 'name': 'label_1'}
    out_path = str(tmp_path / 'test.h5')
    out = layer_writer(out_path, test_layer_data, test_single_data, 'labels', sparse = True)
    assert out == out_path
    assert os.path.isfile(out_path)
