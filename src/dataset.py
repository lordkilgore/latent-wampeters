import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class FaceDataset(Dataset):
    def __init__(self, 
                 img_dir : str, 
                 transforms : nn.Sequential):
        pass