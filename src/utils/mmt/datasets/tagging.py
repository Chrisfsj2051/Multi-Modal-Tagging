import json
import random
from copy import deepcopy

import numpy as np
import torch

from src.utils.mmt import DATASETS
from src.utils.mmt import Compose
from src.utils.mmt.utils import get_root_logger
from src.utils.mmt.utils import calculate_gap


@DATASETS.register_module()
class TaggingDataset:
    def __init__(self, ann_file, label_id_file, pipeline, test_mode=False,
                 with_sigmoid=True):
        (self.index_to_tag, self.tag_to_index, self.index_to_super_index,
         self.tag_to_super_index) = self.load_label_dict(label_id_file)
        self.test_mode = test_mode
        (self.video_anns, self.audio_anns, self.image_anns, self.text_anns,
         self.gt_label, self.gt_onehot) = self.load_annotations(ann_file)
        self.flag = np.zeros((len(self.video_anns))).astype(np.int)
        self.pipeline = Compose(pipeline)
        self.with_sigmoid = with_sigmoid

    @staticmethod
    def load_label_dict(dict_file):
        index_to_tag = {}
        tag_to_index = {}
        index_to_super_index = {}
        tag_to_super_index = {}
        with open(dict_file, 'r', encoding='utf-8') as f:
            contents = f.readlines()
        for i, line in enumerate(contents):
            line = line.strip()
            assert '\t' in line
            tag, classes = line.split('\t')
            index, super_index = classes.split(' ')
            index, super_index = int(index), int(super_index)
            index_to_tag[index] = tag
            tag_to_index[tag] = index
            index_to_super_index[index] = super_index
            tag_to_super_index[tag] = super_index

        return (index_to_tag, tag_to_index, index_to_super_index,
                tag_to_super_index)

    def load_annotations(self, ann_file):
        with open(ann_file, 'r', encoding='utf-8') as f:
            anns = f.readlines()
        modal_anns = [[] for _ in range(6)]
        for _ in range(0, len(anns), 6):
            for i in range(5):
                modal_anns[i].append(anns[_ + i].strip())

        for i in range(len(modal_anns[4])):
            if modal_anns[4][i] == '':
                continue
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
                           text_anns=self.text_anns[i])
            if not self.test_mode:
                results['gt_labels'] = self.gt_label[i]
            results = self.pipeline(deepcopy(results))
            if results is not None:
                return results
            logger = get_root_logger()
            logger.info('Load failed')
            i = random.randint(0, len(self) - 1)

    def format_results(self, outputs, save_dir='submit/submit.json', **kwargs):
        outputs = [x['fusion'][0] for x in outputs]
        outputs = torch.cat([x[None] for x in outputs])
        if self.with_sigmoid:
            outputs = outputs.sigmoid()
        topk_score, topk_label = outputs.topk(20, 1, True, True)
        ret_json = {}
        for i in range(len(self.video_anns)):
            item_id = self.video_anns[i].split('/')[-1].replace('.npy', '.mp4')
            label = [self.index_to_tag[x.item()] for x in topk_label[i]]
            score = [f'{x:.2f}' for x in topk_score[i].tolist()]
            ret_json[item_id] = {
                'result': [{
                    'labels': label,
                    'scores': score
                }]
            }
        with open(save_dir, 'w', encoding='utf-8') as f:
            json.dump(ret_json, f, ensure_ascii=False, indent=4)
            print(f'Saved at {save_dir}')

    def evaluate(self, preds, metric=None, logger=None):
        results = {}
        gt_onehot = np.array(self.gt_onehot)
        for modal in preds[0].keys():
            modal_preds = [x[modal][0] for x in preds]
            if self.with_sigmoid:
                modal_preds = np.array([x.sigmoid().tolist() for x in modal_preds])
            else:
                modal_preds = np.array([x.tolist() for x in modal_preds])
            results[f'{modal}_GAP'] = calculate_gap(modal_preds, gt_onehot)
        print_str = '=' * 30 + '\n'
        print_str += f'{"Modal":8}|{"Super Cls":12}|{"GAP":8}\n'
        super_class = np.unique(list(self.tag_to_super_index.values()))
        for modal in preds[0].keys():
            print_str += '-' * 30 + '\n'
            modal_preds = [x[modal][0] for x in preds]
            if self.with_sigmoid:
                modal_preds = np.array([x.sigmoid().tolist() for x in modal_preds])
            else:
                modal_preds = np.array([x.tolist() for x in modal_preds])
            num_classes = len(self.index_to_super_index)
            for sc in super_class:
                mask = [
                    self.index_to_super_index[x] == sc
                    for x in range(num_classes)
                ]
                mask = np.array(mask)
                sc_modal_preds = modal_preds * mask[None]
                sc_gt_onehot = gt_onehot * mask[None]
                gap = calculate_gap(sc_modal_preds, sc_gt_onehot)
                print_str += f'{modal:8}|{str(sc):12}|{gap:.4f}\n'

        print_str += '-' * 30 + '\n'
        logger = get_root_logger()
        logger.info('\n' + print_str)

        return results

    def __len__(self):
        return len(self.video_anns)


@DATASETS.register_module()
class TaggingDatasetWithAugs(TaggingDataset):
    def __init__(self, strong_pipeline, **kwargs):
        super(TaggingDatasetWithAugs, self).__init__(**kwargs)
        self.strong_pipeline = Compose(strong_pipeline)

    def __getitem__(self, i):
        while True:
            results = dict(audio_anns=self.audio_anns[i],
                           video_anns=self.video_anns[i],
                           image_anns=self.image_anns[i],
                           text_anns=self.text_anns[i])
            if not self.test_mode:
                results['gt_labels'] = self.gt_label[i]
            weak_results = self.pipeline(deepcopy(results))
            strong_results = self.strong_pipeline(deepcopy(results))
            if results is not None and strong_results is not None:
                return {'weak': weak_results, 'strong': strong_results}
            logger = get_root_logger()
            logger.info('Load failed')
            i = random.randint(0, len(self) - 1)


if __name__ == '__main__':
    ds = TaggingDataset(
        ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
        label_id_file='dataset/tagging/label_super_id.txt',
        pipeline=[])
    onehot_list = np.array(ds.gt_onehot)
    onehot = np.sum(onehot_list, 0) / len(onehot_list)
    alpha = (1 - onehot) ** 0.75
    for x in alpha:
        print(x, end=',')
