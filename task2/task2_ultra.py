import os
import time
import copy
import math
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
IMG_SIZE = 384 # EfficientNetV2-S works well with larger resolution
BATCH_SIZE = 16 # Smaller batch size for larger resolution
LEARNING_RATE = 2e-4
EPOCHS = 80 # Increased epochs
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EMA_DECAY = 0.999 # Exponential Moving Average decay

print(f"Using device: {DEVICE}")

# Reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

# --- EMA Class ---
class ModelEMA:
    def __init__(self, model, decay):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + model_v * (1. - self.decay))

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

# --- Data Transforms ---
train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((400, 400)),
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)), # More aggressive cropping
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(45),
    v2.TrivialAugmentWide(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.1),
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

# Dataset
full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")

# Split
# Using 90/10 split again
indices = torch.randperm(len(full_dataset)).tolist()
split = int(0.9 * len(full_dataset))
train_idx = indices[:split]
val_idx = indices[split:]

train_subset = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_transform), train_idx)
val_subset = torch.utils.data.Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_transform), val_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Model: EfficientNetV2-S
print("Initializing EfficientNetV2-S...")
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
# Adjust classifier
# EfficientNetV2 classifier is a Sequential(Dropout, Linear)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)
model = model.to(DEVICE)

# Initialize EMA
ema_model = ModelEMA(model, EMA_DECAY)

# Optimizer & Scheduler
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05) # Higher weight decay for regularization

# Warmup + Cosine Scheduler
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Training Loop
print("Starting training...")
best_f1 = 0.0
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    # Warmup
    if epoch < 5:
        lr = LEARNING_RATE * (epoch + 1) / 5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        # Cosine Decay
        progress = (epoch - 5) / (EPOCHS - 5)
        lr = LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # MixUp/CutMix
        if torch.rand(1).item() < 0.6: 
             images, labels = cutmix_or_mixup(images, labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Update EMA
        ema_model.update(model)
        
        running_loss += loss.item()
        
    # Validation using EMA model
    # EMA model is usually more stable and generalizes better
    ema_model.ema.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = ema_model.ema(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} Val F1 (EMA): {val_f1:.4f} LR: {get_lr(optimizer):.6f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(ema_model.ema.state_dict(), 'best_model_ultra.pth')

print(f"Best Validation F1 (EMA): {best_f1:.4f}")

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

# Load Best EMA Model
model = models.efficientnet_v2_s(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('best_model_ultra.pth'))
model = model.to(DEVICE)
model.eval()

predictions = []
filenames = []

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(DEVICE)
        
        # TTA: Original + Horizontal Flip + Vertical Flip
        # EfficientNetV2 is robust, let's do 3-view TTA
        
        # 1. Original
        out1 = model(images)
        prob1 = torch.softmax(out1, dim=1)
        
        # 2. HFlip
        out2 = model(v2.functional.hflip(images))
        prob2 = torch.softmax(out2, dim=1)
        
        # 3. VFlip (Plants can be viewed from top, rotation matters)
        out3 = model(v2.functional.vflip(images))
        prob3 = torch.softmax(out3, dim=1)
        
        avg_probs = (prob1 + prob2 + prob3) / 3
        _, preds = torch.max(avg_probs, 1)
        
        predictions.extend(preds.cpu().numpy())
        filenames.extend(names)

predicted_labels = [class_names[p] for p in predictions]

submission_df = pd.DataFrame({
    'ID': filenames,
    'Category': predicted_labels
})

submission_df.to_csv('submission_ultra.csv', index=False)
print("Submission saved to submission_ultra.csv")
