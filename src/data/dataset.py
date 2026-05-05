import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random

class FlickrDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def process_image(self, image):

        if image.mode != 'RGB':
            image=image.convert('RGB')

        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = self.process_image(item["image"])
        encoded_captions = item["encoded_caption"]

        if isinstance(encoded_captions[0], list):
            caption = random.choice(encoded_captions)
        else:
            caption = encoded_captions

        caption = torch.tensor(caption, dtype=torch.long)

        return image, caption
    
class BlipDataset(Dataset):
    def __init__(self, raw_data):

        # raw_data: Chính là tập train_data từ dataset nlp/flickr30k chưa qua hàm prepare_data()
        self.data = raw_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"] # Giữ nguyên định dạng PIL Image
        
        # Một ảnh có 5 caption, chọn ngẫu nhiên 1 caption để train cho mỗi epoch
        # (Cách này giúp model đa dạng hóa vốn từ)
        captions = item["caption"]
        if isinstance(captions, list):
            caption = random.choice(captions)
        else:
            caption = captions
            
        # Trả về chuỗi nguyên thủy thay vì vector số
        return image, caption

# Vì dataloader mặc định của PyTorch không biết cách gộp (batch) các object kiểu PIL Image,
# ta phải viết hàm collate_fn tùy chỉnh:
def blip_collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return images, texts