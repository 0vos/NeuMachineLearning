import os
import cv2
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

# --- Configuration ---
DATA_DIR = 'NeuMachineLearning-main/neu-plant-seedling-classification-num2-2025/dataset-for-task2'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Hyperparameters for "Ultra" performance
IMG_SIZE = 384  # Increased resolution (Standard SOTA is often 384 or 448)
BATCH_SIZE = 16 # Reduced batch size to fit larger images in memory
LEARNING_RATE = 1e-4 # Slightly lower LR for fine-tuning
EPOCHS = 40     # EfficientNet converges faster
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

print(f"Using device: {DEVICE}")

# --- Reproducibility ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True 

set_seed(SEED)

# --- Preprocessing: Green Masking (Segmentation) ---
# This implements your "Segmentation" idea. 
# It removes background noise (soil, stones) by keeping only green pixels.
def segment_plant(image_np):
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # Define range of green color in HSV
    # These values are tuned for plant seedlings
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    
    # Create a mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(image_np, image_np, mask=mask)
    
    return result

class SegmentedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # Apply segmentation before transforms
        sample_np = np.array(sample)
        sample_segmented = segment_plant(sample_np)
        sample = Image.fromarray(sample_segmented)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return sample, target

# --- Transforms ---
# Stronger augmentations for the larger model
train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((400, 400)), # Resize slightly larger
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)), # Stronger cropping
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(45),
    v2.TrivialAugmentWide(), # AutoAugment
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.1), # Lower probability since we already mask background
])

val_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MixUp / CutMix
cutmix = v2.CutMix(num_classes=5)
mixup = v2.MixUp(num_classes=5)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

# --- Dataset & Loaders ---
# Use our custom SegmentedImageFolder
full_dataset = SegmentedImageFolder(root=TRAIN_DIR, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")

# 90/10 Split
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
indices = torch.randperm(len(full_dataset)).tolist()
train_idx = indices[:train_size]
val_idx = indices[train_size:]

# We need separate datasets to apply different transforms
train_subset = torch.utils.data.Subset(SegmentedImageFolder(TRAIN_DIR, transform=train_transform), train_idx)
val_subset = torch.utils.data.Subset(SegmentedImageFolder(TRAIN_DIR, transform=val_transform), val_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# --- Model: EfficientNetV2-S ---
# EfficientNetV2 is faster and more accurate than V1 and DenseNet.
# "S" (Small) is comparable to ResNet50 but much better.
print("Initializing EfficientNetV2-S...")
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

# Adjust classifier
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)
model = model.to(DEVICE)

# --- Training Setup ---
# Label Smoothing helps prevent overconfidence
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

# Cosine Annealing with Warm Restarts (helps escape local minima)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# --- Training Loop ---
print("Starting training...")
best_f1 = 0.0
best_model_path = 'best_model_boost.pth'

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # MixUp/CutMix
        if torch.rand(1).item() < 0.5:
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
        torch.save(model.state_dict(), best_model_path)

print(f"Best Validation F1: {best_f1:.4f}")

# --- Inference with TTA (Test Time Augmentation) ---
print("Generating predictions with TTA...")

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
        
        # Apply segmentation to test images too!
        image_np = np.array(image)
        image_segmented = segment_plant(image_np)
        image = Image.fromarray(image_segmented)
        
        if self.transform:
            image = self.transform(image)
        return image, img_name

test_dataset = TestDataset(root_dir=TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model.load_state_dict(torch.load(best_model_path))
model.eval()

predictions = []
filenames = []

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(DEVICE)
        
        # TTA: Average predictions of Original + Horizontal Flip + Vertical Flip
        # 1. Original
        out1 = model(images)
        prob1 = torch.softmax(out1, dim=1)
        
        # 2. Horizontal Flip
        images_h = v2.functional.hflip(images)
        out2 = model(images_h)
        prob2 = torch.softmax(out2, dim=1)
        
        # 3. Vertical Flip
        images_v = v2.functional.vflip(images)
        out3 = model(images_v)
        prob3 = torch.softmax(out3, dim=1)
        
        # Average
        avg_probs = (prob1 + prob2 + prob3) / 3
        _, preds = torch.max(avg_probs, 1)
        
        predictions.extend(preds.cpu().numpy())
        filenames.extend(names)

predicted_labels = [class_names[p] for p in predictions]

submission_df = pd.DataFrame({
    'ID': filenames,
    'Category': predicted_labels
})

submission_df.to_csv('submission_boost.csv', index=False)
print("Submission saved to submission_boost.csv")
