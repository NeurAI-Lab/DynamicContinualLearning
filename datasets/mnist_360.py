# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from datasets.transforms.rotation import IncrementalRotation
from argparse import Namespace
import numpy as np
from PIL import Image
from copy import deepcopy
from datasets.utils.validation import get_train_val
from datasets.seq_mnist import MyMNIST as MyMNISTBase


class MyMNIST(MyMNISTBase):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.ToTensor()
        super(MyMNIST, self).__init__(root, train,
                                      transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(torch.stack((img, img, img), dim=-1).numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img, target, img
        else:
            return img, target


class MNIST360:
    """
    MNIST-360 general continual dataset.
    """
    NAME = 'mnist-360'
    SETTING = 'general-continual'
    N_CLASSES = 9
    LENGTH = 54051

    def __init__(self, args: Namespace) -> None:
        self.num_rounds = 3
        self.args = args
        self.train_over, self.test_over = False, False

        self.train_loaders, self.test_loaders = [], []
        self.remaining_training_items = []
        self.val_dataset = None

        self.train_classes = [0, 1]
        self.completed_rounds, self.test_class, self.test_iteration = 0, 0, 0

        self.init_train_loaders(base_data_path=args.data_path)
        self.init_test_loaders(base_data_path=args.data_path)

        self.active_train_loaders = [
            self.train_loaders[self.train_classes[0]].pop(),
            self.train_loaders[self.train_classes[1]].pop()]

        self.active_remaining_training_items = [
            self.remaining_training_items[self.train_classes[0]].pop(),
            self.remaining_training_items[self.train_classes[1]].pop()]

    def train_next_class(self) -> None:
        """
        Changes the couple of current training classes.
        """
        self.train_classes[0] += 1
        self.train_classes[1] += 1
        if self.train_classes[0] == self.N_CLASSES: self.train_classes[0] = 0
        if self.train_classes[1] == self.N_CLASSES: self.train_classes[1] = 0

        if self.train_classes[0] == 0:
            self.completed_rounds += 1
            if self.completed_rounds == 3:
                self.train_over = True

        if not self.train_over:
            self.active_train_loaders = [
                self.train_loaders[self.train_classes[0]].pop(),
                self.train_loaders[self.train_classes[1]].pop()]
            self.active_remaining_training_items = [
                self.remaining_training_items[self.train_classes[0]].pop(),
                self.remaining_training_items[self.train_classes[1]].pop()]

    def init_train_loaders(self, base_data_path) -> None:
        """
        Initializes the test loader.
        """
        train_dataset = MyMNIST(base_data_path + 'MNIST',
                                train=True, download=True)
        if self.args.validation:
            test_transform = transforms.ToTensor()
            train_dataset, self.val_dataset = get_train_val(
                train_dataset, test_transform, self.NAME)

        for j in range(self.N_CLASSES):
            self.train_loaders.append([])
            self.remaining_training_items.append([])
            train_mask = np.isin(np.array(train_dataset.targets), [j])
            train_rotation = IncrementalRotation(init_deg=(j - 1) * 60,
                            increase_per_iteration=360.0 / train_mask.sum())
            for k in range(self.num_rounds * 2):
                tmp_train_dataset = deepcopy(train_dataset)
                numbers_per_batch = train_mask.sum() // (self.num_rounds * 2) + 1
                tmp_train_dataset.data = tmp_train_dataset.data[
                    train_mask][k * numbers_per_batch:(k+1) * numbers_per_batch]
                tmp_train_dataset.targets = tmp_train_dataset.targets[
                    train_mask][k * numbers_per_batch:(k+1) * numbers_per_batch]
                tmp_train_dataset.transform = transforms.Compose(
                    [train_rotation, transforms.ToTensor()])
                self.train_loaders[-1].append(DataLoader(
                    tmp_train_dataset, batch_size=1, shuffle=True))
                self.remaining_training_items[-1].append(
                    tmp_train_dataset.data.shape[0])

    def init_test_loaders(self, base_data_path) -> None:
        """
        Initializes the test loader.
        """
        if self.args.validation:
            test_dataset = self.val_dataset
        else:
            test_dataset = MyMNIST(base_data_path + 'MNIST',
                                 train=False, download=True)
        for j in range(self.N_CLASSES):
            tmp_test_dataset = deepcopy(test_dataset)
            test_mask = np.isin(np.array(tmp_test_dataset.targets), [j])
            tmp_test_dataset.data = tmp_test_dataset.data[test_mask]
            tmp_test_dataset.targets = tmp_test_dataset.targets[test_mask]
            test_rotation = IncrementalRotation(
                increase_per_iteration=360.0 / test_mask.sum())
            tmp_test_dataset.transform = transforms.Compose(
                [test_rotation, transforms.ToTensor()])
            self.test_loaders.append(DataLoader(tmp_test_dataset,
                            batch_size=self.args.batch_size, shuffle=True))

    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ensembles the next examples of the current classes in a single batch.
        :return: the augmented and not aumented version of the examples of the
                 current batch, along with their labels.
        """
        assert not self.train_over
        batch_size_0 = min(int(round(self.active_remaining_training_items[0] /
                                     (self.active_remaining_training_items[0] +
                                      self.active_remaining_training_items[1]) *
                                     self.args.batch_size)),
                           self.active_remaining_training_items[0])

        batch_size_1 = min(self.args.batch_size - batch_size_0,
                           self.active_remaining_training_items[1])

        x_train, y_train, x_train_naug = [], [], []
        for j in range(batch_size_0):
            i_x_train, i_y_train, i_x_train_naug = next(iter(
                self.active_train_loaders[0]))
            x_train.append(i_x_train)
            y_train.append(i_y_train)
            x_train_naug.append(i_x_train_naug)
        for j in range(batch_size_1):
            i_x_train, i_y_train, i_x_train_naug = next(iter(
                self.active_train_loaders[1]))
            x_train.append(i_x_train)
            y_train.append(i_y_train)
            x_train_naug.append(i_x_train_naug)
        x_train, y_train, x_train_naug = torch.cat(x_train),\
                                    torch.cat(y_train), torch.cat(x_train_naug)

        self.active_remaining_training_items[0] -= batch_size_0
        self.active_remaining_training_items[1] -= batch_size_1

        if self.active_remaining_training_items[0] <= 0 or \
                self.active_remaining_training_items[1] <= 0:
            self.train_next_class()

        return x_train, y_train, x_train_naug

    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensembles the next examples of the current class in a batch.
        :return: the batch of examples along with its label.
        """
        assert not self.test_over
        x_test, y_test = next(iter(self.test_loaders[self.test_class]))
        residual_items = len(self.test_loaders[self.test_class].dataset) - \
                        self.test_iteration * self.args.batch_size - len(x_test)
        self.test_iteration += 1
        if residual_items <= 0:
            if residual_items < 0:
                x_test = x_test[:residual_items]
                y_test = y_test[:residual_items]
            self.test_iteration = 0
            self.test_class += 1
            if self.test_class == self.N_CLASSES:
                self.test_over = True
        return x_test, y_test

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None