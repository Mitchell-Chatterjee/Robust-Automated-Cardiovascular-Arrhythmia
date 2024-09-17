from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.core.datasets.ecg_interface import Split


class DataLoaders:
    def __init__(
            self,
            datasetCls,
            dataset_kwargs: dict,
            batch_size: int,
            workers: int = 0,
            reset_strat_folds: bool = False,
            collate_fn=None,
            shuffle_train=True,
            shuffle_val=False,
            distributed=True
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size

        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.reset_strat_folds = reset_strat_folds
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val
        self.distributed = distributed

        self.train, self.train_sampler = self.train_dataloader()
        self.valid, self.valid_sampler = self.val_dataloader()
        self.test, self.test_sampler = self.test_dataloader()

    def train_dataloader(self):
        return self._make_dloader(Split.TRAIN, reset_strat_folds=self.reset_strat_folds, shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self._make_dloader(Split.VALIDATION, reset_strat_folds=False, shuffle=self.shuffle_val)

    def test_dataloader(self):
        return self._make_dloader(Split.TEST, reset_strat_folds=False, shuffle=False)

    def _make_dloader(self, split, reset_strat_folds, shuffle=False):
        dataset = self.datasetCls(split=split, reset_strat_folds=reset_strat_folds, **self.dataset_kwargs)
        sampler = DistributedSampler(dataset) if self.distributed else None
        if len(dataset) == 0:
            return None, None

        if self.distributed:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
                shuffle=False,
                sampler=sampler
            ), sampler
        else:
            return DataLoader(
                dataset,
                shuffle=shuffle,
                batch_size=self.batch_size,
                num_workers=self.workers,
                collate_fn=self.collate_fn,
            ), sampler
