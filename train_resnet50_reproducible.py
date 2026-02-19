import torch
from torchvision import models, transforms
import logging
import os
import numpy as np
from sklearn.metrics import accuracy_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hyperparameters
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MANUAL_SEED = 42

# Set seed for reproducibility
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset (example using CIFAR10)
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load the ResNet50 model
model = models.resnet50(pretrained=True)

# Modify the final layer for our specific task
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)  # Assuming 10 classes for CIFAR10
model = model.to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
train_losses = []
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    epoch_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    logging.info(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}')

# Saving the model
model_save_path = './resnet50_model.pth'
torch.save(model.state_dict(), model_save_path)
logging.info(f'Model saved to {model_save_path}')

# Overfitting analysis
# Define validation dataset and loader here for analysis...
# Log the accuracy on validation and test sets after training

# Check performance
# validation_accuracy = accuracy_score(...)
# logging.info(f'Validation Accuracy: {validation_accuracy:.2f}')