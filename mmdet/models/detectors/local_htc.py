from ..builder import DETECTORS
from .local_cascade import LocalCascadeRCNN


@DETECTORS.register_module()
class LocalHybridTaskCascade(LocalCascadeRCNN):
    """Implementation of `HTC <https://arxiv.org/abs/1901.07518>`_"""

    def __init__(self, **kwargs):
        super(LocalHybridTaskCascade, self).__init__(**kwargs)

    @property
    def with_semantic(self):
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic
