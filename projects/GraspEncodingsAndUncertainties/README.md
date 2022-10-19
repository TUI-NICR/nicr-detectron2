# On the Importance of Label Encoding and Uncertainty Estimation for Robotic Grasp Detection

This file describes the usage of this repository for our paper "On the Importance of Label Encoding and Uncertainty Estimation for Robotic Grasp Detection".

This file only describes how to train models and asumes you already prepared the datasets as described [here](https://github.com/TUI-NICR/nicr-grasping).

The training for models with aleatoric uncertainty estimation slightly differs from other models, as it is advantageous to use pretrained weights for a more stable training.

## Training

Normal models (without uncertainty estimation) can easily be trained by using the `tools/train_grasp.py`.

Example:
```bash
python tools/train_grasp.py --config config/graspnet_grconvnet_depth.yaml
```
For a more detailed tutorial for using detectron2 for training see the [official documentation](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html).

Wether to use our proposed loss masking is decided through the `MODEL.GRASP_NET.MASK_LOSS` config parameter.
Similarly the `MODEL.GRASP_NET.INVERT_MASK_LOSS` decides wether to use the inner ignore or outer ignore scheme.

The provided config files assume a `GRConvNet` but if you want to train a `GGCNN2` you can set `MODEL.META_ARCHITECTURE` to `GGCNN2` instead.

### Training with uncertainty (epistemic)

As epistemic uncertainty is estimated by using MC-Dropout you only need to train a model with dropout.
For this a simple config is also supplied and can be used instead.

Example:
```bash
python tools/train_grasp.py --config config/graspnet_grconvnet_depth_uncertainty.yaml
```

### Training with uncertainty (aleatoric)

For training a model which, in addition to the epistemic uncertainty, also estimates the aleatoric uncertainy it is recommended to use a pretained network.
This means training a model with dropout but without estimating aleatoric uncertainy.
Based on a set of pretrained weights the meta architeture `UncertainGraspNet` can be used to extend the model with aleatoric uncertainty estimation. The pretrained weights need to be specified through the `UNCERTAIN_NET_WEIGHTS` parameter NOT the `MODEL.WEIGHTS` parameter, as the latter would try to load a pretrained model already equipped with aleatoric estimation.

Example:
```bash
python tools/train_grasp.py --config config/graspnet_grconvnet_depth_al_uncertainty.yaml MODEL.GRASP_NET.UNCERTAIN_NET_WEIGHTS <PATH_TO_YOUR_PRETRAINED_WEIGHTS>
```

## Model weights and configs

A selection of our best models mentioned in the paper are available through the links below (GoogleDrive).

### Models without uncertainty estimation
| Model | Config | Weights |
| --- | --- | --- |
| Gauss with outer ignore | [Config](configs/gauss_outer_ignore.yaml) | [Weights](https://drive.google.com/uc?id=1VrCYL1IOHyDCpRA4JuKstgf94t36GLD7) |

### Models with epistemic uncertainty estimation

These models include a dropout layer with dropout rate set to 0.2.
This is necessary for using MC-Dropout during inference.

| Model | Config | Weights |
| --- | --- | --- |
| Gauss with outer ignore | [Config](configs/gauss_outer_ignore_ep.yaml) | [Weights](https://drive.google.com/uc?id=1TQzcKpuIP8zrGBFu7fT22BVlYlPZ1FI6) |
| Gauss with inner ignore | [Config](configs/gauss_inner_ignore_ep.yaml) | [Weights](https://drive.google.com/uc?id=16A47wZckNWOIPEcKbAjltmqZZlDmc2OG) |

### Model with aleatoric and epistemic uncertainty
This model was trained with the weights from `Gauss with inner ignore` in the section above as pretraining.

| Model | Config | Weights |
| --- | --- | --- |
| Gauss with inner ignore | [Config](configs/gauss_inner_ignore_ep_al.yaml) | [Weights](https://drive.google.com/uc?id=1F2QsXyx875SowvX9Wo1TZGpTi5yyqQ5V) |
