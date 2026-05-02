import torch
from torch.utils.data import Dataset
import numpy as np

class FlickrDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def process_image(self, image):
        image = image.resize((224, 224))
        image = np.array(image, dtype=np.float32) / 255.0
        return torch.tensor(image).permute(2, 0, 1)  # (C, H, W)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = self.process_image(item["image"])
        caption = item["encoded_caption"]

        return image, caption