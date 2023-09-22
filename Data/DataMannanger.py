
import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class MultiLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # Extracting folder names as classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # Mapping class names to indices
        self.image_paths = []

        for cls in self.classes:
            for image_name in os.listdir(os.path.join(root_dir, cls)):
                self.image_paths.append((os.path.join(root_dir, cls, image_name), cls))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx][0]
        image = Image.open(img_name).convert('RGB')  # Ensure the image is in RGB format
        
        # Check if the transform is not None before applying it
        if self.transform:
            image = self.transform(image)
        
        # Get the labels for the image
        label_name = self.image_paths[idx][1]
        label_idx = self.class_to_idx[label_name]
        
        # Convert the label index into a one-hot encoded vector
        label_vector = [0] * len(self.classes)
        label_vector[label_idx] = 1
        
        return image, torch.tensor(label_vector, dtype=torch.float32)