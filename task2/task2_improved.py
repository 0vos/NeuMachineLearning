import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets, models
from sklearn.metrics import f1_score
from PIL import Image
import time

# Configuration
DATA_DIR = 'NeuMachineLearning-main/neu-plant-seedling-classification-num2-2025/dataset-for-task2'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
IMG_SIZE = 224  # Increased to 224 for standard pre-trained models
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # Lower learning rate for fine-tuning
EPOCHS = 50  # Increased epochs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

print(f"Using device: {DEVICE}")

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Transforms
# Enhanced data augmentation for small dataset
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
class_names = full_dataset.classes
print(f"Classes: {class_names}")

# Split into Train and Validation
# Using indices to split cleanly
indices = list(range(len(full_dataset)))
np.random.shuffle(indices)
split = int(np.floor(0.2 * len(full_dataset)))
train_idx, val_idx = indices[split:], indices[:split]

# We need separate datasets to apply different transforms
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=val_transform)

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

# Increased num_workers for faster data loading
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4, pin_memory=True)

# Test Dataset
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

test_dataset = TestDataset(root_dir=TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Model - Using ResNet50
print("Initializing ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Replace the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, len(class_names))
)

model = model.to(DEVICE)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
# Using AdamW which often generalizes better
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

# Training Loop
print("Starting training...")
best_f1 = 0.0
patience_counter = 0
early_stopping_patience = 15

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    epoch_time = time.time() - start_time
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}, Time: {epoch_time:.2f}s")
    
    # Step the scheduler
    scheduler.step(val_f1)
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
        print(f"  New best model saved! F1: {best_f1:.4f}")
    else:
        patience_counter += 1
        
    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

print(f"Best Validation F1 Score: {best_f1:.4f}")

# Prediction
print("Generating predictions...")
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

predictions = []
filenames = []

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        predictions.extend(preds.cpu().numpy())
        filenames.extend(names)

# Map predictions back to class names
predicted_labels = [class_names[p] for p in predictions]

# Create submission DataFrame
submission_df = pd.DataFrame({
    'ID': filenames,
    'Category': predicted_labels
})

# Save submission
submission_df.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
