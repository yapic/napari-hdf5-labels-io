from napari_hdf5_labels_io._writer import write_layers_h5, layer_writer, compress_layer, process_metadata
import numpy as np
import os

test_meta = {'key1': 0, 'metadata': {'a': 0, 'b': 1}, 'key3': 2}
out_shape = (10, 25, 25)
test_layer_data = np.random.randint(6, size=out_shape)
sparse_array = np.array([[[0, 1, 0],
                          [1, 0, 1],
                          [1, 0, 0]],
                         [[0, 1, 1],
                          [0, 1, 1],
                          [1, 1, 0]]])

def test_process_metadata():
    out_dict = {'key1': 0, 'meta_a': 0, 'meta_b': 1, 'key3': 2}
    assert process_metadata(test_meta) == out_dict


def test_compress_layer():
    shape, data, is_sparse = compress_layer(sparse_array)
    assert is_sparse == True
    assert data.shape[0] == 4
    assert data.shape[1] == 10

    shape, data, is_sparse = compress_layer(test_layer_data)
    assert out_shape == shape
    if is_sparse:
         assert len(data) == 4
         assert data.shape[1] == test_layer_data.sum()
    else:
        assert (data[:] == test_layer_data).all()

def test_layer_writer(tmp_path):
    test_single_data = {'key1': 0, 'metadata': {'a': 0, 'b': 1}, 'data': [1, 2, 3], 'name': 'label_1'}
    out_path = str(tmp_path / 'test.h5')
    out = layer_writer(out_path, test_layer_data, test_single_data, 'labels')
    assert out == out_path
    assert os.path.isfile(out_path)
