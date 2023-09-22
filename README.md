# MultiLabelCNNClassifier

## Introduction
`MultiLabelCNNClassifier` is a CNN-based framework designed to streamline the process of training custom classifiers. With a straightforward setup and intuitive directory structure, you can easily prepare your image data and start training.

## Directory Structure
To get started, place your image data into the `Data/Images` directory. The expected structure is as follows:

```
Data
└── Images
    ├── AniName.jpg
    ├── AniName.jpg
    ├── AniName.jpg
    └── AniName.jpg
```

## Data Augmentation
After placing your images in the appropriate directory, run the `helper.py` script within the `Data` directory using the command:

```
py helper.py
```

This script aids in balancing your dataset through data augmentation. Data augmentation is a technique used to artificially increase the size of your dataset by creating modified versions of existing data. This ensures a balanced training process, which is crucial for achieving optimal machine learning model performance.

## Model Configuration
If you wish to modify the default settings of the model, navigate to the `MODEL SETTINGS` section in `agent.py`. Here, you can adjust parameters such as the training epoch. Additionally, you can configure whether the CNN skips the training phase and proceeds directly to testing.

## Visualizing Neural Network Filters
To view the neural network filters, execute the `extract-filters` script located in the main directory. This will extract and visualize the model's filters. If you've changed the model's name, ensure you update it within this script to correctly track the model.

## Customization and Exploration
This framework is designed to be both robust and flexible. If you're experienced, you're encouraged to modify and adapt it to your needs. And even if you're a beginner, don't hesitate to experiment with different settings and values. Through exploration, you'll undoubtedly gain valuable insights and knowledge. Happy coding, and may the force be with you!
