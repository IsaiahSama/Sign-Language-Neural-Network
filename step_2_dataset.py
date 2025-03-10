from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch

from typing import Tuple, List

import csv


class SignLanguageMNIST(Dataset):
    """Sign Language Classification Dataset.

    This class is a utility class for loading the Sign Language Dataset into PyTorch.

    """

    @staticmethod
    def get_label_mapping() -> list[int]:
        """
        This method is used to map all labels to [0, 23].

        Returns: Mapping from dataset labels [0, 23] to letter indices [0, 25]

        Out of the 26 letters, J and Z are excluded.
        """

        mapping = list(range(25))
        mapping.pop(9)  # Removes the 9th value of J
        return mapping

    @staticmethod
    def read_label_samples_from_csv(path: str) -> Tuple[list, list]:
        """Reads the label samples from the provided CSV file.

        Args:
            path (str): The path of the CSV file to load.

        This method assumes the first column in CSV is the label, and the subsequent values are image pixel values from 0-255.
        """

        mapping = SignLanguageMNIST.get_label_mapping()
        labels, samples = [], []

        with open(path) as f:
            reader = csv.reader(f)
            _ = next(reader)  # Getting the header fields

            for row in reader:
                label = int(row[0])
                labels.append(mapping.index(label))
                samples.append(list(map(int, row[1:])))

        return labels, samples

    def __init__(
        self,
        path: str = "data/sign_mnist_train.csv",
        mean: List[float] = [0.485],
        std: List[float] = [0.229],
    ):
        """Method to initialize the data holder.

        Args:
            path (str, optional): The path to the training data. Defaults to "data/sign_mnist_train.csv".
            mean (List[float], optional): The mean. Defaults to [0.485].
            std (List[float], optional): The value to use for the standard deviation. Defaults to [0.229].
        """

        labels, samples = SignLanguageMNIST.read_label_samples_from_csv(path)
        self._samples = np.array(samples, dtype=np.uint8).reshape((-1, 28, 28, 1))
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1, 1))
        
        self._mean = mean
        self._std = std

    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self, index):
        """Returns a dictionary containing the sample and the label
        
        Uses a technique called data augmentation, where samples are preturbed during training to increase model's robusness.
        
        In this case, will randomly zoom in on the image by varying amounts and on different locations via RandomResizedCrop.
        
        We then normalize the inputs so that the image values are rescaled to the [0, 1] range in expectation.
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)
        ])

        return {
            'image': transform(self._samples[index]).float(),
            'label': torch.from_numpy(self._labels[index]).float()
        }