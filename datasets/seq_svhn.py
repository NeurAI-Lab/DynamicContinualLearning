from torchvision.datasets import SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple


class MySVHN(SVHN):
    """
    Overrides the MNIST dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.ToTensor()
        self.train = train
        split = "train" if train else "test"
        super(MySVHN, self).__init__(root, split,
                                      transform, target_transform, download)
        self.targets = self.labels
        #self.seen_track = [False] * 5

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        # if not self.train:
        #     if not self.seen_track[target // 5]:
        #         import matplotlib.pyplot as plt
        #         plt.imshow(np.array(img))
        #         plt.show()
        #         self.seen_track[target // 5] = True
        original_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train and hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        if self.train:
            return img, target, original_img
        else:
            return img, target


class SequentialSVHN(ContinualDataset):

    NAME = 'seq-svhn'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = None

    def get_data_loaders(self, base_data_path):
        transform = transforms.ToTensor()
        train_dataset = MySVHN(base_data_path + 'SVHN',
                             train=True,
                             download=False, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        transform, self.NAME)
        else:
            test_dataset = MySVHN(base_data_path + 'SVHN',
                                  train=False,
                                  download=False, transform=transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size, base_data_path):
        transform = transforms.ToTensor()
        train_dataset = MySVHN(base_data_path + 'SVHN',
                                train=True, download=True, transform=transform)
        train_mask = np.logical_and(np.array(train_dataset.targets) >= self.i -
            self.N_CLASSES_PER_TASK, np.array(train_dataset.targets) < self.i)

        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True)
        return train_loader

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None
