import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models
from torchvision.transforms import v2
from sklearn.metrics import f1_score
from PIL import Image

# Configuration
DATA_DIR = 'NeuMachineLearning-main/neu-plant-seedling-classification-num2-2025/dataset-for-task2'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Hyperparameters
IMG_SIZE = 384
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 70
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

print(f"Using device: {DEVICE}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torch.__version__}")

# Reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

# --- Data Transforms using Torchvision V2 ---
# V2 transforms are faster and support MixUp/CutMix natively

# Define MixUp/CutMix
# We will apply these inside the training loop or collate_fn, 
# but v2 recommends applying them as part of the transform pipeline if possible,
# however, MixUp requires batch processing. 
# The standard way in v2 is to use them in a custom collate function or after loading a batch.

# Base transforms for individual images
train_transform = v2.Compose([
    v2.ToImage(), # Convert to tensor image
    v2.Resize((280, 280)),
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(30),
    v2.TrivialAugmentWide(), # State-of-the-art automatic augmentation
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.2),
])

val_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MixUp / CutMix Wrapper
# We need to apply this to a BATCH of data, not single images.
cutmix = v2.CutMix(num_classes=5)
mixup = v2.MixUp(num_classes=5)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return v2.functional.collate(batch)

# Dataset
full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")

# Split Train/Val (Single Split for Speed, but high ratio for training)
# 90% Train, 10% Val - We need as much data as possible for training to reach 0.95
# We will rely on the test set submission for final verification if val is noisy.
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Fix validation transform
# random_split doesn't allow changing transform easily. 
# We'll just use the train_transform for validation but without the random parts? 
# No, that's bad. We need clean validation.
# Let's re-instantiate.
indices = torch.randperm(len(full_dataset)).tolist()
train_idx = indices[:train_size]
val_idx = indices[train_size:]

train_subset = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_transform), train_idx)
val_subset = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_transform), val_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Model: DenseNet121
# DenseNet is excellent for small datasets due to feature reuse.
print("Initializing DenseNet121...")
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, num_classes)
model = model.to(DEVICE)

# Optimizer & Scheduler
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training Loop
print("Starting training...")
best_f1 = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Apply MixUp/CutMix
        if torch.rand(1).item() < 0.5: # 50% chance to apply mixup/cutmix
             images, labels = cutmix_or_mixup(images, labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    scheduler.step()
    
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
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} Val F1: {val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), 'best_model_final.pth')

print(f"Best Validation F1: {best_f1:.4f}")

# --- Inference ---
print("Generating predictions...")

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
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model.load_state_dict(torch.load('best_model_final.pth'))
model.eval()

predictions = []
filenames = []

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(DEVICE)
        
        # TTA: Original + Horizontal Flip
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        
        images_flip = v2.functional.hflip(images)
        outputs_flip = model(images_flip)
        probs_flip = torch.softmax(outputs_flip, dim=1)
        
        avg_probs = (probs + probs_flip) / 2
        _, preds = torch.max(avg_probs, 1)
        
        predictions.extend(preds.cpu().numpy())
        filenames.extend(names)

predicted_labels = [class_names[p] for p in predictions]

submission_df = pd.DataFrame({
    'ID': filenames,
    'Category': predicted_labels
})

submission_df.to_csv('submission_final384size.csv', index=False)
print("Submission saved to submission_final384size.csv")