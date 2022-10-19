import os

import torch
import torch.optim as optim

from detectron2.data import (
    build_detection_train_loader,
    DatasetCatalog,
)
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import DatasetEvaluators

from detectron2.solver.build import get_default_optimizer_params
from detectron2.data.samplers.distributed_sampler import InferenceSampler
from detectron2.data.common import MapDataset

from nicr_grasping import graspnet_dataset_path

from nicr_detectron2.evaluation.grasp_evaluator import (
    GraspVisualizationEvaluator, GraspNetPredictionSaver
)

from nicr_detectron2.data.graspnet_mapper import (
    GraspDatasetMapper as DatasetMapper
)


class GraspNetTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        train_datasets = cfg.DATASETS.TRAIN
        assert len(train_datasets) > 0, "No train dataset given"
        assert len(train_datasets) == 1, "Too many train datasets!"
        return build_detection_train_loader(
            DatasetCatalog.get(train_datasets[0]),
            mapper=DatasetMapper(cfg, is_train=True),  # pylint: disable=redundant-keyword-arg,missing-kwoa
            total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_workers=cfg.DATALOADER.NUM_WORKERS)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # return build_detection_test_loader(
        #     DatasetCatalog.get(dataset_name),
        #     mapper=DatasetMapper(cfg, is_train=False),
        #     num_workers=cfg.DATALOADER.NUM_WORKERS)
        def trivial_batch_collator(batch):
            """
            A batch collator that does nothing.
            """
            return batch
        dataset = DatasetCatalog.get(dataset_name)
        mapper = DatasetMapper(cfg, is_train=False)  # pylint: disable=redundant-keyword-arg,missing-kwoa
        total_batch_size = cfg.SOLVER.IMS_PER_BATCH
        # random.seed(1337)
        #samples = random.sample(list(range(len(dataset))), 64)
        #dataset = [dataset[sample] for sample in samples]
        sampler = InferenceSampler(len(dataset))
        num_workers = cfg.DATALOADER.NUM_WORKERS
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler,
            int(total_batch_size),
            drop_last=False)
        data_loader = torch.utils.data.DataLoader(
            MapDataset(dataset, mapper),
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
        )
        return data_loader

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # TODO: make this parametrizable through config or args
        if any('graspnet' in d for d in cfg.DATASETS.TEST):
            # return GraspVisualizationEvaluator(dataset_name=dataset_name)
            return DatasetEvaluators([
                GraspVisualizationEvaluator(cfg, dataset_name=dataset_name),
                GraspNetPredictionSaver(cfg, graspnet_dataset_path())
            ])

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators(
            [
                GraspVisualizationEvaluator(cfg, dataset_name=dataset_name)
            ])

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        """
        params = get_default_optimizer_params(model)
        optimizer = optim.Adam(params)
        return optimizer
