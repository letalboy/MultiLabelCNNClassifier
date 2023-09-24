import numpy as np
import matplotlib.pyplot as plt

import torch
import os

from agent import MultiLabelClassifier, RCNN_MultiLabelClassifier

THIS_DIR = os.path.dirname(__file__)
MODELS = os.path.join(THIS_DIR, "Models")
MODEL = os.path.join(MODELS, "Test1.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MultiLabelClassifier(num_labels=11).to(device)

model = MultiLabelClassifier(num_labels=15).to(device)

model.load_state_dict(torch.load(MODEL))
model.eval()

# Extract weights of the first and second convolutional layers
conv1_weights = model.features[0].weight.data.cpu().numpy()
conv2_weights = model.features[3].weight.data.cpu().numpy()

# Function to normalize an array between 0 and 1
def normalize(arr):
    arr_min, arr_max = np.min(arr), np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)

# Function to plot filters
def plot_filters(weights, num_columns, title):
    num_kernels = weights.shape[0]
    num_rows = int(num_kernels / num_columns)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_kernels:
            ax.imshow(normalize(weights[i][0]), cmap='gray')
            ax.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.show()

# Plot filters of the first and second convolutional layers
plot_filters(conv1_weights, 8, 'Filters of Conv1 Layer')
plot_filters(conv2_weights, 8, 'Filters of Conv2 Layer')
