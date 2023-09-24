import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from torchvision.io import read_image
import os
from PIL import Image
import torch.nn.functional as F

class MultiLabelClassifier(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, num_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class RCNN_MultiLabelClassifier(nn.Module):
    class RCL(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, num_iterations):
            super().__init__()

            # Feed-forward convolution
            self.conv_feed_forward = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)

            # Recurrent convolution
            self.conv_recurrent = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)

            # Number of recurrent iterations
            self.num_iterations = num_iterations

        def forward(self, x):
            # Initial feed-forward pass
            out = self.conv_feed_forward(x)

            # Recurrent iterations
            for _ in range(self.num_iterations):
                out = F.relu(self.conv_feed_forward(x) + self.conv_recurrent(out))

            return out
        
    def __init__(self, num_labels, num_iterations=2, device=torch.device("cpu")):
        super(RCNN_MultiLabelClassifier, self).__init__()
        
        # Integrate the RCNN structure
        self.rcnn_features = nn.Sequential(
            # First convolutional layer remains feed-forward
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Four RCLs with a max-pooling layer in the middle
            self.RCL(64, 64, 3, num_iterations).to(device),
            self.RCL(64, 64, 3, num_iterations).to(device),
            nn.MaxPool2d(3, stride=2, padding=1),
            self.RCL(64, 64, 3, num_iterations).to(device),
            self.RCL(64, 64, 3, num_iterations).to(device),
            
            # Global max-pooling layer
            nn.AdaptiveMaxPool2d(1)
        )
        
        # Classifier remains the same but with adjusted input size
        self.classifier = nn.Sequential(
            nn.Linear(64, 1024),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, num_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.rcnn_features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

