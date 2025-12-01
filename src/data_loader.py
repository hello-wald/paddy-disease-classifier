from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from config import config 

class RiceDiseaseDataset(Dataset):
    """Custom dataset for rice disease classification"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform

        self.classes = config.CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.[jJ][pP][gG]'): # .jpg or .JPG
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
                    
        print(f"Found {len(self.images)} images in {len(set(self.labels))} classes")
        for i, class_name in enumerate(self.classes):
            count = sum(1 for label in self.labels if label == i)
            print(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label