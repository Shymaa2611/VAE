from torch.utils.data import  Dataset
from PIL import Image
import os
from glob import glob

class UnsupervisedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob(os.path.join(root_dir, '*'))  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  
        if self.transform:
            image = self.transform(image)
        return image, 0  

