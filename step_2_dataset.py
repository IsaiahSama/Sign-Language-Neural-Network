from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch

from typing import Tuple

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
        mapping.pop(9) # Removes the 9th value of J
        return mapping

    @staticmethod
    def read_label_samples_from_csv(path:str) -> Tuple[list, list]:
        """Reads the label samples from the provided CSV file.

        Args:
            path (str): The path of the CSV file to load.
            
        This method assumes the first column in CSV is the label, and the subsequent values are image pixel values from 0-255.
        """
        
        mapping = SignLanguageMNIST.get_label_mapping()
        labels, samples = [], []
        
        with open(path) as f:
            reader = csv.reader(f)
            _ = next(reader) # Getting the header fields
            
            for row in reader:
                label = int(row[0])
                labels.append(mapping.index(label))
                samples.append(list(map(int, row[1:])))

        return labels, samples
            
