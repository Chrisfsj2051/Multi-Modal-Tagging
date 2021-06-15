import random
from copy import deepcopy

import numpy as np

from mmt.datasets import TaggingDataset
from mmt.datasets.builder import DATASETS
from mmt.utils import get_root_logger
from mmt.utils.metrics.calculate_gap import calculate_gap


def get_another(x, low, high):
    y = x
    while x == y:
        y = random.randint(low, high)
    return y


@DATASETS.register_module()
class PretrainMatchDataset(TaggingDataset):

    def __init__(self, **kwargs):
        super(PretrainMatchDataset, self).__init__(**kwargs)
        self.match_list = ([(i, i) for i in range(len(self.video_anns))] +
                           [(i, get_another(i, 0, len(self.video_anns) - 1))
                            for i in range(len(self.video_anns))])
        self.gt_onehot = [int(self.match_list[i][1] == self.match_list[i][0])
                          for i in range(len(self.match_list))]
        self.flag = np.zeros((len(self))).astype(np.int)

    def __getitem__(self, i):
        while True:
            results = dict(
                audio_anns=self.audio_anns[self.match_list[i][0]],
                video_anns=self.video_anns[self.match_list[i][0]],
                image_anns=self.image_anns[self.match_list[i][0]],
                text_anns=self.test_anns[self.match_list[i][1]])
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
        for modal in preds[0].keys():
            modal_preds = [x[modal][0] for x in preds]
            modal_preds = np.array([x.sigmoid().tolist() for x in modal_preds])
            t = (modal_preds >= 0.5).squeeze().astype(np.int)
            results[f'{modal}_accuracy'] = sum([x == y for x, y in zip(gt_onehot, t)]) / len(gt_onehot)

        return results

    def __len__(self):
        return len(self.match_list)
