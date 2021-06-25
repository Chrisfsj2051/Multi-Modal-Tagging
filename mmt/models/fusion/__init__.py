from .base import BaseFusionModel
from .multi_branch import MultiBranchFusionModel
from .single_branch import SingleBranchModel
from .modal_match import ModalMatchModel
from .video_text_extra_task import VideoTextWithExtraTaskModel

__all__ = ['BaseFusionModel', 'MultiBranchFusionModel', 'SingleBranchModel',
           'ModalMatchModel', 'VideoTextWithExtraTaskModel']
