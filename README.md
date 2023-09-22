# MultiLabelCNNClassifier

## Introduction
`MultiLabelCNNClassifier` is a CNN-based framework designed to streamline the process of training custom classifiers. With a straightforward setup and intuitive directory structure, you can easily prepare your image data and start training.

## Directory Structure
To get started, place your image data into the `Data/Images` directory. The expected structure is as follows:

```
Data
└── Images
        ├── Category
        |    ├── AniName.jpg
        |    ├── AniName.jpg
        |    ├── AniName.jpg
        |    └── AniName.jpg
        └── Category
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

## Model Training
Once you've set up your presets, you can begin training your model by executing `agent.py`. The system will save a checkpoint in the "Models" directory every 10 epochs. If you wish to retain any of these checkpoints, simply transfer them to a different folder. After doing so, you can utilize the saved model whenever you like. However, ensure that you use the same agent configurations each time. For instance, if you trained your model with 10 classes, you must run it with those same 10 classes.

## Model Evaluation

To evaluate the performance of your model, follow these steps:

1. Disable the training mode.
2. Create a folder named "Tests" and place your test images inside.
3. Configure the test section accordingly.
4. Run the evaluation. If set up correctly, the system will process the images from the "Tests" folder and provide predictions based on the trained model. Ensure you review the results to verify the model's accuracy and effectiveness in making predictions.

## Visualizing Neural Network Filters
To view the neural network filters, execute the `extract-filters` script located in the main directory. This will extract and visualize the model's filters. If you've changed the model's name, ensure you update it within this script to correctly track the model.

## Customization and Exploration
This framework is designed to be both robust and flexible. If you're experienced, you're encouraged to modify and adapt it to your needs. And even if you're a beginner, don't hesitate to experiment with different settings and values. Through exploration, you'll undoubtedly gain valuable insights and knowledge. Happy coding, and may the force be with you!
