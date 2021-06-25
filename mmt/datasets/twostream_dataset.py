from .builder import DATASETS, build_dataset
import random

@DATASETS.register_module()
class TwoStreamDataset:
    def __init__(self, main_dataset_config, extra_dataset_config):
        self.main_dataset = build_dataset(main_dataset_config)
        self.extra_dataset = build_dataset(extra_dataset_config)

    def __len__(self):
        return len(self.main_dataset)

    def __getitem__(self, x):
        main_data = self.main_dataset[x]
        extra_data = self.extra_dataset[random.randint(0, len(self.extra_dataset))]
        main_data['extra'] = extra_data
        return main_data