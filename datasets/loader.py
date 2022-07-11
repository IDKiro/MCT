import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        filename_A = self.files_A[index % len(self.files_A)]
        item_A = self.transform(Image.open(filename_A).convert(mode='RGB'))

        if self.unaligned:
            filename_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            filename_B = self.files_B[index % len(self.files_B)]

        item_B = self.transform(Image.open(filename_B).convert(mode='RGB'))

        return {'A': item_A, 'filename_A':os.path.basename(filename_A), 
                'B': item_B, 'filename_B':os.path.basename(filename_B)}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
