import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import os
from PIL import Image
import torch.distributed as dist

from Data.DataMannanger import MultiLabelDataset
from model import MultiLabelClassifier, RCNN_MultiLabelClassifier

# > PREDEFINED CONSTs

THIS_DIR = os.path.dirname(__file__)
MODELS = os.path.join(THIS_DIR, "Models")
MODEL = os.path.join(MODELS, "test1.pth")

# > MODEL SETTINGS

NUM_OF_LABELS = 15
TRAINING = True
TRAINING_EPOCHS = 100

# Create the directory if it doesn't exist
if not (os.path.exists(MODELS)):
    os.mkdir(MODELS)

transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = MultiLabelDataset(root_dir=os.path.join(THIS_DIR, 'Data', 'Augmented_Images'), transform=transform)

def main_worker(gpu, ngpus_per_node):

    # Initialize the distributed environment
    dist.init_process_group(backend='gloo', init_method='tcp://localhost:12345', world_size=ngpus_per_node, rank=gpu)

    # Set up the device and move the model to the GPU
    torch.cuda.set_device(gpu)
    model = MultiLabelClassifier(num_labels=NUM_OF_LABELS).to(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Adjust batch size based on GPU (assuming GPU 0 has more memory)
    batch_size = 32 if gpu == 0 else 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    if os.path.exists(MODEL) and gpu == 0:
        # Load the state dictionary from the file
        state_dict = torch.load(MODEL)

        # Check if the model is an instance of DistributedDataParallel
        if isinstance(model, nn.parallel.DistributedDataParallel):
            # Add 'module.' prefix to each key
            state_dict = {'module.' + k: v for k, v in state_dict.items()}

        # Load the modified state dictionary into the model
        model.load_state_dict(state_dict)

        model.eval()

    if TRAINING:
        print(f"Training on GPU: {gpu}")
        for epoch in range(TRAINING_EPOCHS):
            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(gpu, non_blocking=True), labels.cuda(gpu, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

            print(f"GPU: {gpu} | Epoch {epoch+1}/{TRAINING_EPOCHS} Loss: {loss.item()}")

        # Save the model (only from GPU 0 to avoid conflicts)
        if gpu == 0:
            torch.save(model.module.state_dict(), MODEL)

def main():
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))

# > Test your model

def classify_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        outputs = model(image)

    # Post-process the output (assuming a multi-label classification problem)
    predicted_labels = (outputs > 0.5).squeeze().cpu().numpy()  # Convert output probabilities to binary labels

    # Convert binary labels to class names
    class_names = train_dataset.classes  # Assuming train_dataset is available and has the 'classes' attribute
    predicted_class_names = [class_names[i] for i, label in enumerate(predicted_labels) if label]

    return predicted_class_names

if __name__ == '__main__':
    # Recommended for Windows, though not strictly necessary for this script
    mp.freeze_support()
    
    if TRAINING:
        main()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = MultiLabelClassifier(num_labels=NUM_OF_LABELS).to(device)
        model.load_state_dict(torch.load(MODEL))
        model.eval()

        image_path = "Tests/Test3.jpg"
        print(classify_image(image_path))

