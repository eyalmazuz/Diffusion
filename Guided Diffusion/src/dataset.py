import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Normalize, Compose, RandomHorizontalFlip

class ImageDataset(Dataset):

    def __init__(self, image_path, image_size) -> None:
        super().__init__()

        self.path = image_path
        self.images = os.listdir(image_path)

        self.image_size = image_size
        self.compose = Compose([ToTensor(), Resize(image_size), Normalize((0.5), (0.5)), RandomHorizontalFlip(0.3)])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> str:
        image = cv2.imread(os.path.join(self.path, self.images[idx]))
        image = self.compose(image)

        return image