import random
from copy import deepcopy

import numpy as np

from src.utils.mmt import TaggingDataset
from src.utils.mmt import DATASETS
from src.utils.mmt.utils import get_root_logger


def get_another(x, low, high):
    y = x
    while x == y:
        y = random.randint(low, high)
    return y


@DATASETS.register_module()
class ModalMatchDataset(TaggingDataset):

    def __init__(self, **kwargs):
        super(ModalMatchDataset, self).__init__(**kwargs)
        self.match_list = ([(i, i) for i in range(len(self.video_anns))] +
                           [(i, get_another(i, 0, len(self.video_anns) - 1))
                            for i in range(len(self.video_anns))])
        self.gt_onehot = [int(self.match_list[i][1] == self.match_list[i][0])
                          for i in range(len(self.match_list))]
        self.flag = np.zeros((len(self))).astype(np.int)

    def __getitem__(self, i):
        while True:
            # if not self.test_mode and self.gt_onehot[i] == 0:
            #     self.match_list[i] = (self.match_list[i][0],
            #                           get_another(self.match_list[i][0], 0, len(self.video_anns)-1))
            results = dict(
                audio_anns=self.audio_anns[self.match_list[i][0]],
                video_anns=self.video_anns[self.match_list[i][0]],
                image_anns=self.image_anns[self.match_list[i][0]],
                text_anns=self.text_anns[self.match_list[i][1]],
                meta_info={'modal_match': True}
            )
            if not self.test_mode:
                results['gt_labels'] = [self.gt_onehot[i]]
            results = self.pipeline(deepcopy(results))
            if results is not None:
                return results
            logger = get_root_logger()
            logger.info('Load failed')
            i = random.randint(0, len(self) - 1)

    def evaluate(self, preds, metric=None, logger=None):
        results = {}
        gt_onehot = np.array(self.gt_onehot)
        modal_preds = np.array([x.sigmoid().tolist() for x in preds])
        t = (modal_preds >= 0.5).squeeze().astype(np.int)
        results[f'accuracy'] = sum([x == y for x, y in zip(gt_onehot, t)]) / len(gt_onehot)
        results['preds_mean'] = round(modal_preds.mean(), 4)

        return results

    def __len__(self):
        # return 20
        return len(self.match_list)
