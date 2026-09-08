from torch.utils.data import Dataset
from world import make_world


class MicroVLADataset(Dataset):

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image, target = make_world()

        return image, target