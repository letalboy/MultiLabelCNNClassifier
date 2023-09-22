import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
import os
from PIL import Image

from Data.DataMannanger import MultiLabelDataset
from model import MultiLabelClassifier

# > PREDEFINED CONSTs

THIS_DIR = os.path.dirname(__file__)
MODELS = os.path.join(THIS_DIR, "Models")
MODEL = os.path.join(MODELS, "test1.pth")

# > MODEL SETTINGS

# Change this to skip training if you alwready ahve a trained model

TRAINING = True
TRAINING_EPOCHS = 100

# >-------------------------------------------------------------------------------------------------------------------------------------------------------
# > Code:

if not (os.path.exists(MODELS)):
    os.mkdir(MODELS)

transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = MultiLabelDataset(root_dir=os.path.join(THIS_DIR, 'Data', 'Augmented_Images'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiLabelClassifier(num_labels=11).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if os.path.exists(MODEL):
    model.load_state_dict(torch.load(MODEL))
    model.eval()

if TRAINING:
    for epoch in range(TRAINING_EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{TRAINING_EPOCHS} Loss: {loss.item()}")

        torch.save(model.state_dict(), MODEL)

# > -----------------------------------------------------------------------------------------------------------------------------------------------------
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

image_path = "Tests/Test3.jpg"
print(classify_image(image_path))