import numpy as np
import h5py
from napari_hdf5_labels_io._reader import h5_to_napari
from napari_hdf5_labels_io._writer import compress_layer

# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    my_test_file = str(tmp_path / "myfile.h5")
    label_data = np.random.choice((0, 1), (20, 20), p=[0.9, 0.1])
    original_shape, compressed_data, is_sparse = compress_layer(label_data)

    with h5py.File(my_test_file, 'w') as hdf:
        labels = hdf.create_group('labels')
        label1 = labels.create_dataset('1abel1', data=compressed_data)
        label1.attrs['shape'] = original_shape
        label1.attrs['pos'] = 0
        label1.attrs['is_sparse'] = is_sparse

    # try to read it back in
    reader = h5_to_napari(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # make sure it's the same as it started
    np.testing.assert_allclose(label_data, layer_data_tuple[0])


def test_get_reader_pass():
    reader = h5_to_napari("fake.file")
    assert reader is None
