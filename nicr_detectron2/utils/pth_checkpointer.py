from collections import OrderedDict
from detectron2.checkpoint import DetectionCheckpointer

# make a small change in DetectionCheckpointer to make loading of .pth files available for nested models
class PTHDetectionCheckpointer(DetectionCheckpointer):
    def __init__(self, model, prefix=""):
        super().__init__(model)
        self.prefix = prefix

    # just add the matching_heuristics field and let the super implementation do the work
    def _load_model(self, checkpoint):
        checkpoint.update({"matching_heuristics": True})
        current_state_dict = checkpoint["model"]
        new_state_dict = OrderedDict()
        for key in current_state_dict.keys():
            new_state_dict[self.prefix+key] = current_state_dict[key]
        checkpoint["model"] = new_state_dict
        return super()._load_model(checkpoint)
