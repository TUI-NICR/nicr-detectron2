from detectron2.config import CfgNode as CN


def add_multimodal_config(cfg):
    cfg.INPUT.MODALITIES = ()
    cfg.INPUT.FORMAT = []
    cfg.INPUT.MAX_DEPTH = 0

    cfg.MODEL.PIXEL_MEANS = []
    cfg.MODEL.PIXEL_STDS = []

    return cfg


def add_grasp_config(cfg):

    cfg = add_multimodal_config(cfg)

    # augmentation
    cfg.INPUT.ROTATE = False
    cfg.INPUT.ZOOM = False

    # UncertainGraspNet
    cfg.MODEL.GRASP_NET = CN()

    # evaluator
    cfg.MODEL.GRASP_NET.VISUALIZE_EVAL = False

    # uncertain grasp net
    cfg.MODEL.GRASP_NET.UNCERTAIN_NET = "GGCNN2"
    cfg.MODEL.GRASP_NET.UNCERTAIN_NET_WEIGHTS = ""

    # smoothed gnlll/l1
    cfg.MODEL.GRASP_NET.USE_SMOOTHED_LOSS = False
    cfg.MODEL.GRASP_NET.MAX_LOSS = False
    cfg.MODEL.GRASP_NET.MAX_LOSS_UPDATE_PERIOD = 100
    cfg.MODEL.GRASP_NET.MASK_LOSS = False
    cfg.MODEL.GRASP_NET.INVERT_MASK_LOSS = False
    cfg.MODEL.GRASP_NET.GAUSS_POSTPROCESS = False
    cfg.MODEL.GRASP_NET.VARIANCE_HEAD = False

    cfg.MODEL.GRASP_NET.MC_DROPOUT = False
    cfg.MODEL.GRASP_NET.MC_SAMPLES = 50
    cfg.MODEL.GRASP_NET.MAX_UNCERTAINTY = 1.0

    # parameter for local maximum search
    # min quality for recovered grasps
    cfg.MODEL.GRASP_NET.MIN_QUALITY = 0.01
    # min distance between grasps in pixels
    cfg.MODEL.GRASP_NET.MIN_DISTANCE = 1
    # Maximum of estimated grasps
    cfg.MODEL.GRASP_NET.MAX_GRASP_NUM = 200

    # grconvnet
    cfg.MODEL.GRASP_NET.GRCONVNET = CN()
    cfg.MODEL.GRASP_NET.GRCONVNET.DROPOUT_RATE = 0.0
    cfg.MODEL.GRASP_NET.GRCONVNET.NUM_RES_BLOCKS = 5
    cfg.MODEL.GRASP_NET.GRCONVNET.TYPE_RES_BLOCK = "ResidualBlock"

    # GGCNN2
    cfg.MODEL.GRASP_NET.GGCNN2 = CN()
    cfg.MODEL.GRASP_NET.GGCNN2.DROPOUT_RATE = 0.0

    return cfg
