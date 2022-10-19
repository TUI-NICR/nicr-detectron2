from logging import root
import torchvision.datasets as datasets
from detectron2.data import DatasetCatalog, MetadataCatalog
import json
import os


# input is a dataset dict
# {
#   "input_files": [{"color_image":"path/to/img", "depth_image":"path/to/img"},{...},...],
#   "label_files": [{"quality":"path","angle":"path","width":"path"},{}]
# }
def load_dataset_from_json(json_path):
    from . import DATASET_PATH
    dataset = []
    root_dir = DATASET_PATH
    with open(os.path.join(DATASET_PATH, json_path), "r") as f:
        input_dict = json.load(f)
        root_dir = os.path.dirname(os.path.join(DATASET_PATH, json_path))
        for input_files, label_files in zip(input_dict["input_files"], input_dict["label_files"]):
            for key in input_files.keys():
                input_files[key] = os.path.join(root_dir, input_files[key]).replace("npy","npz") # TODO: remove hotfix
            for key in label_files.keys():
                label_files[key] = os.path.join(root_dir, label_files[key]).replace("npy","npz")
            dataset.append(
                {
                    "input_files": input_files,
                    "label_files": label_files
                }
            )
        return dataset

def register_classification_dataset(name, metadata, json_path):

    assert isinstance(name, str), name

    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_dataset_from_json(json_path))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        evaluator_type="grasp_evaluator", **metadata
    )

cornell_train_path = "cornell/train.json"
cornell_test_path = "cornell/test.json"

graspnet_train_path = "graspnet/train.json"
graspnet_test_path = "graspnet/test.json"
graspnet_test_seen_path = "graspnet/test_seen.json"
graspnet_test_similar_path = "graspnet/test_similar.json"
graspnet_test_novel_path = "graspnet/test_novel.json"

# register into DatasetCatalog
register_classification_dataset("cornell_train", {"dataset_type": "grasping"}, cornell_train_path)
register_classification_dataset("cornell_test", {"dataset_type": "grasping"}, cornell_test_path)

register_classification_dataset("graspnet_train", {"dataset_type": "grasping"}, graspnet_train_path)
register_classification_dataset("graspnet_test", {"dataset_type": "grasping"}, graspnet_test_path)

register_classification_dataset("graspnet_test_seen", {"dataset_type": "grasping"}, graspnet_test_seen_path)
register_classification_dataset("graspnet_test_similar", {"dataset_type": "grasping"}, graspnet_test_similar_path)
register_classification_dataset("graspnet_test_novel", {"dataset_type": "grasping"}, graspnet_test_novel_path)

print("Registered the grasping datasets!")
