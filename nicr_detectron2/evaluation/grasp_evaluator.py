import os
import numpy as np
import torch

from torchvision.transforms.functional import resize as resize_pt
from torchvision.transforms.functional import pad

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import MetadataCatalog
from detectron2.utils.events import get_event_storage

from nicr_grasping.utils.postprocessing import convert_model_output_to_grasps
from nicr_grasping.datatypes.grasp_conversion import CONVERTER_REGISTRY

from graspnetAPI.grasp import RectGraspGroup, GraspGroup
from graspnetAPI import GraspNetEval

from ..utils.visualizer_grasp import GraspVisualizer


class GraspVisualizationEvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name) -> None:
        super().__init__()
        self.img_found = False
        self.metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._fig_dir = os.path.join(cfg.OUTPUT_DIR, 'example_predictions')
        os.makedirs(self._fig_dir, exist_ok=True)

    def process(self, inputs, outputs):
        if not self.img_found:
            for inpt, output in zip(inputs, outputs):
                if 'color' in inpt['image']:
                    image = inpt['image']['color']
                else:
                    image = inpt['image'][list(inpt['image'].keys())[0]]
                storage = get_event_storage()
                vis = GraspVisualizer(
                    np.moveaxis(
                        image.cpu().numpy(),
                        0, -1)
                    .squeeze()
                )
                f, _ = vis.visualize_prediction(output, inpt)
                f.savefig(os.path.join(self._fig_dir, 'exam_prediction.png'))
                f.canvas.draw()
                vis_img = np.frombuffer(f.canvas.tostring_rgb(), dtype=np.uint8)
                vis_img = vis_img.reshape(f.canvas.get_width_height()[::-1] + (3,))
                # vis_image = vis_img.get_image().transpose(2,0,1)
                storage.put_image(
                    'Example Prediction',
                    vis_img.astype(np.uint8).transpose(2, 0, 1))
                self.img_found = True
                break

    def evaluate(self):
        self.img_found = False
        return


class GraspNetPredictionSaver(DatasetEvaluator):
    def __init__(self, cfg, graspnet_root:str, camera : str='kinect') -> None:
        super().__init__()
        self._cpu_device = torch.device("cpu")
        self._graspnet = GraspNetEval(graspnet_root, camera, 'all')
        self._save_dir_base = os.path.join(cfg.OUTPUT_DIR, 'graspnet_predictions')

        self._min_quality = cfg.MODEL.GRASP_NET.MIN_QUALITY
        self._min_distance = cfg.MODEL.GRASP_NET.MIN_DISTANCE
        self._max_grasp_num = cfg.MODEL.GRASP_NET.MAX_GRASP_NUM
        self._max_uncertainty = cfg.MODEL.GRASP_NET.MAX_UNCERTAINTY

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for inp, outp in zip(inputs, outputs):
            scene_id = inp['sample_id'] // 256
            ann_id = inp['sample_id'] % 256

            maps = [outp['pos'], outp['ang'], outp['width']]

            h, w = inp['original_shape'][:2]
            shortest_edge = min(h, w)

            padd_x = h - shortest_edge
            padd_y = w - shortest_edge

            # reconstruct original resolution
            # needed for properly projecting grasps into 3D as pixel coordinates
            # need to be in original images
            # TODO: This may be faster through transforming found grasps
            pos = resize_pt(
                outp["pos"], (shortest_edge, shortest_edge))
            ang = resize_pt(
                outp["ang"], (shortest_edge, shortest_edge))
            width = resize_pt(
                outp["width"], (shortest_edge, shortest_edge))
            pos = pad(pos, (padd_y // 2, padd_x // 2))
            ang = pad(ang, (padd_y // 2, padd_x // 2))
            width = pad(width, (padd_y // 2, padd_x // 2))

            additional_maps = None

            if 'aleatoric' in outp:
                uncerts = outp['aleatoric']
                add_maps = {'pos_var': uncerts['pos'],
                            'sin_var': uncerts['sin'],
                            'cos_var': uncerts['cos'],
                            'width_var': uncerts['width']}
                # TODO: refactor map postprocessing as function
                add_maps = {
                    key: resize_pt(m, (shortest_edge, shortest_edge))
                    for key, m in add_maps.items()
                }
                add_maps = {
                    key: pad(m, (padd_y // 2, padd_x // 2))
                    for key, m in add_maps.items()
                }

                if additional_maps is None:
                    additional_maps =  add_maps
                else:
                    additional_maps.update(add_maps)
            if 'epistemic'  in outp:
                uncerts = outp['epistemic']
                add_maps = {'pos_ep_var': uncerts['pos'],
                            'sin_ep_var': uncerts['sin'],
                            'cos_ep_var': uncerts['cos'],
                            'width_ep_var': uncerts['width']}
                add_maps = {
                    key: resize_pt(m, (shortest_edge, shortest_edge))
                    for key, m in add_maps.items()
                }
                add_maps = {
                    key: pad(m, (padd_y // 2, padd_x // 2))
                    for key, m in add_maps.items()
                }

                if additional_maps is None:
                    additional_maps =  add_maps
                else:
                    additional_maps.update(add_maps)

            masks = self._graspnet.loadMask(scene_id, self._graspnet.camera, ann_id)
            workspace_mask = masks != 0
            workspace_mask = workspace_mask.astype(np.uint8)
            workspace_mask_torch = torch.from_numpy(workspace_mask).to(pos.device)

            # quality, angle, width = maps
            pos *= workspace_mask_torch.unsqueeze(0)

            maps = (pos, ang, width)

            grasps = convert_model_output_to_grasps(maps,
                                                    min_quality=self._min_quality,
                                                    num_grasps=self._max_grasp_num,
                                                    min_distance=self._min_distance,
                                                    loc_max_version = "skimage_cpu",
                                                    additional_maps=additional_maps)
            if 'epistemic' in outp:
                grasps.grasps = [
                    g for g in grasps.grasps
                    if g._additional_params['pos_ep_var'] + g._additional_params['sin_ep_var'] + g._additional_params['cos_ep_var'] + g._additional_params['width_ep_var'] < self._max_uncertainty]

            os.makedirs(
                os.path.join(self._save_dir_base,
                             f'scene_{scene_id:04d}',
                             self._graspnet.camera),
                exist_ok=True)

            current_save_dir = os.path.join(self._save_dir_base,
                                            f'scene_{scene_id:04d}',
                                            self._graspnet.camera,
                                            f'{ann_id:04d}')

            if len(grasps) != 0:
                grasps.save(os.path.join(self._save_dir_base,
                                         f'scene_{scene_id:04d}',
                                         self._graspnet.camera,
                                         f'{ann_id:04d}_orig.pkl'))

                graspnet_grasps = CONVERTER_REGISTRY.convert(grasps, RectGraspGroup)
                depth_img = self._graspnet.loadDepth(sceneId=scene_id,
                                                     camera=self._graspnet.camera,
                                                     annId=ann_id)

                graspnet_grasps = graspnet_grasps.to_grasp_group(self._graspnet.camera,
                                                                 depth_img)
            else:
                graspnet_grasps = GraspGroup()
            graspnet_grasps.save_npy(current_save_dir)

    def evaluate(self):
        acc = self._graspnet.eval_scene(scene_id=100,
                                        dump_folder=self._save_dir_base,
                                        vis=False,
                                        TOP_K=50)
        return {'AP-scene100': np.mean(acc)}
