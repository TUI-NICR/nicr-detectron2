from nicr_detectron2.data.multimodal_dataset_mapper import DatasetMapper

def test_multimodal_mapper(multimodal_config, multimodal_dataset_sample_dict):
    mapper = DatasetMapper(multimodal_config, is_train=False, augmentations=[])

    sample = mapper(multimodal_dataset_sample_dict)

    assert 'image' in sample
    for mod in multimodal_config.INPUT.MODALITIES:
        shape = sample['image'][mod].shape
        assert shape[1] == sample['height']
        assert shape[2] == sample['width']
