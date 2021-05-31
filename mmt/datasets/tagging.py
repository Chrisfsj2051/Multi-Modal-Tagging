import random

import numpy as np

from mmt.datasets.builder import DATASETS
from mmt.datasets.pipelines import Compose
from mmt.utils.metrics.calculate_gap import calculate_gap


@DATASETS.register_module()
class TaggingDataset:
    def __init__(self, ann_file, label_id_file, pipeline, test_mode=False):
        self.index_to_tag, self.tag_to_index = self.load_label_dict(
            label_id_file)
        (self.video_anns, self.audio_anns, self.image_anns, self.test_anns,
         self.gt_label, self.gt_onehot) = self.load_annotations(ann_file)
        self.flag = np.zeros((len(self.video_anns)))
        self.pipeline = Compose(pipeline)

    def load_label_dict(self, dict_file):
        index_to_tag = {}
        tag_to_index = {}
        with open(dict_file, 'r', encoding='utf-8') as f:
            contents = f.readlines()
        for i, line in enumerate(contents):
            line = line.strip()
            if '\t' in line:
                index, tag = line.split('\t')[:2]
            elif ' ' in line:
                index, tag = i, line.rsplit(' ', 1)[0]
            else:
                index, tag = i, line

            try:
                index = int(index)
            except Exception:
                index, tag = int(tag), index

            index_to_tag[index] = tag
            tag_to_index[tag] = index
        return index_to_tag, tag_to_index

    def load_annotations(self, ann_file):
        with open(ann_file, 'r', encoding='utf-8') as f:
            anns = f.readlines()
        modal_anns = [[] for _ in range(6)]
        for _ in range(0, len(anns), 6):
            for i in range(5):
                modal_anns[i].append(anns[_ + i].strip())

        for i in range(len(modal_anns[4])):
            line = modal_anns[4][i].split(',')
            modal_anns[4][i] = [self.tag_to_index[x] for x in line]
            modal_anns[5].append(np.zeros((len(self.tag_to_index))))
            modal_anns[5][-1][np.array(modal_anns[4][i])] = 1

        return modal_anns

    def __getitem__(self, i):
        while True:
            results = dict(audio_anns=self.audio_anns[i],
                           video_anns=self.video_anns[i],
                           image_anns=self.image_anns[i],
                           text_anns=self.test_anns[i],
                           gt_labels=self.gt_label[i])
            results = self.pipeline(results)
            if results is not None:
                return results
            i = random.randint(0, len(self) - 1)

    def evaluate(self, preds, logger):
        preds = np.array([x.sigmoid().tolist() for x in preds])
        gt_onehot = np.array(self.gt_onehot)
        gap = calculate_gap(preds, gt_onehot)
        return dict(GAP=gap)

    def __len__(self):
        return len(self.video_anns)


if __name__ == '__main__':
    TaggingDataset('data/tagging/GroundTruth/datafile/train.txt',
                   'data/tagging/label_id.txt', [])
