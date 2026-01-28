import torch
import numpy

from torch.utils.data import Dataset



# Define dataclass for the dataset
class CustomDataset(Dataset):

    def __init__(self, data, color):
        if isinstance(data, numpy.ndarray):
            self.data = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            self.data = data.float()
        else:
            raise TypeError("Data must be a torch.Tensor or numpy.ndarray!")
        self.color = color

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.color[idx], idx