DATASET_PATH = None

from . import graspnet_datasets


def set_datasets_root(root):
    global DATASET_PATH
    DATASET_PATH = root


def get_datasets_root():
    return DATASET_PATH
