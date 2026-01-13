import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, models
from torchvision.transforms import v2
from sklearn.metrics import f1_score
from PIL import Image
import torch.nn.functional as F

# --- Configuration ---
DATA_DIR = 'NeuMachineLearning-main/neu-plant-seedling-classification-num2-2025/dataset-for-task2'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Hyperparameters
IMG_SIZE = 384
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 35 # Slightly reduced as we might run two phases
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
CONFIDENCE_THRESHOLD = 0.90 # Threshold for pseudo-labeling

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

# --- Preprocessing: Green Masking ---
def segment_plant(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(image_np, image_np, mask=mask)
    return result

# --- Custom Datasets ---
class SegmentedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample_np = np.array(sample)
        sample_segmented = segment_plant(sample_np)
        sample = Image.fromarray(sample_segmented)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

class PseudoLabelDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        image_segmented = segment_plant(image_np)
        image = Image.fromarray(image_segmented)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

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
        image_np = np.array(image)
        image_segmented = segment_plant(image_np)
        image = Image.fromarray(image_segmented)
        if self.transform:
            image = self.transform(image)
        return image, img_name

# --- Transforms ---
train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((400, 400)),
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
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

# --- Helper Functions ---
def get_model(num_classes):
    print("Initializing EfficientNetV2-S...")
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model.to(DEVICE)

def train_one_epoch(model, loader, criterion, optimizer, use_mixup=True):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if use_mixup and torch.rand(1).item() < 0.5:
             images, labels = cutmix_or_mixup(images, labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro')

def predict_tta(model, loader):
    model.eval()
    predictions = []
    filenames = []
    probs_list = []
    
    with torch.no_grad():
        for images, names in loader:
            images = images.to(DEVICE)
            
            # TTA: Original + HFlip + VFlip
            out1 = model(images)
            prob1 = torch.softmax(out1, dim=1)
            
            out2 = model(v2.functional.hflip(images))
            prob2 = torch.softmax(out2, dim=1)
            
            out3 = model(v2.functional.vflip(images))
            prob3 = torch.softmax(out3, dim=1)
            
            avg_probs = (prob1 + prob2 + prob3) / 3
            _, preds = torch.max(avg_probs, 1)
            
            predictions.extend(preds.cpu().numpy())
            filenames.extend(names)
            probs_list.extend(avg_probs.cpu().numpy())
            
    return filenames, predictions, np.array(probs_list)

# --- Main Execution ---

# 1. Setup Data
full_dataset = SegmentedImageFolder(root=TRAIN_DIR, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")

# Split
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
indices = torch.randperm(len(full_dataset)).tolist()
train_idx = indices[:train_size]
val_idx = indices[train_size:]

train_subset = torch.utils.data.Subset(SegmentedImageFolder(TRAIN_DIR, transform=train_transform), train_idx)
val_subset = torch.utils.data.Subset(SegmentedImageFolder(TRAIN_DIR, transform=val_transform), val_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 2. Phase 1: Train Teacher (or Load)
teacher_model_path = 'best_model_boost.pth'
model = get_model(num_classes)

if os.path.exists(teacher_model_path):
    print(f"Loading existing teacher model from {teacher_model_path}...")
    model.load_state_dict(torch.load(teacher_model_path))
else:
    print("Teacher model not found. Training from scratch...")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_f1 = 0.0
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        val_f1 = validate(model, val_loader)
        print(f"Teacher Epoch [{epoch+1}/{EPOCHS}] Loss: {loss:.4f} Val F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), teacher_model_path)
    model.load_state_dict(torch.load(teacher_model_path))

# 3. Phase 2: Pseudo-Labeling
print("Generating Pseudo-Labels...")
test_dataset = TestDataset(root_dir=TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

filenames, preds, probs = predict_tta(model, test_loader)

pseudo_images = []
pseudo_labels = []
count = 0

for i, filename in enumerate(filenames):
    max_prob = np.max(probs[i])
    if max_prob > CONFIDENCE_THRESHOLD:
        img_path = os.path.join(TEST_DIR, filename)
        pseudo_images.append(img_path)
        pseudo_labels.append(preds[i])
        count += 1

print(f"Generated {count} pseudo-labels out of {len(filenames)} test images (Threshold: {CONFIDENCE_THRESHOLD})")

# 4. Phase 3: Train Student on Combined Data
print("Starting Student Training with Pseudo-Labels...")

# Create Combined Dataset
pseudo_dataset = PseudoLabelDataset(pseudo_images, pseudo_labels, transform=train_transform)
# Note: We use the original full training set (train + val) for the student to maximize data
# But to monitor progress, we still keep a validation set.
# Strategy: Train on (Train_Subset + Pseudo), Validate on (Val_Subset)
combined_dataset = ConcatDataset([train_subset, pseudo_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# Initialize Student Model (Fresh)
student_model = get_model(num_classes)
student_model_path = 'best_model_pseudo.pth'

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

best_student_f1 = 0.0

for epoch in range(EPOCHS):
    loss = train_one_epoch(student_model, combined_loader, criterion, optimizer)
    scheduler.step()
    
    # Validate on the clean validation set
    val_f1 = validate(student_model, val_loader)
    
    print(f"Student Epoch [{epoch+1}/{EPOCHS}] Loss: {loss:.4f} Val F1: {val_f1:.4f}")
    
    if val_f1 > best_student_f1:
        best_student_f1 = val_f1
        torch.save(student_model.state_dict(), student_model_path)

print(f"Best Student Validation F1: {best_student_f1:.4f}")

# 5. Final Inference
print("Generating Final Submission...")
student_model.load_state_dict(torch.load(student_model_path))

filenames, preds, _ = predict_tta(student_model, test_loader)
predicted_labels = [class_names[p] for p in preds]

submission_df = pd.DataFrame({
    'ID': filenames,
    'Category': predicted_labels
})

submission_df.to_csv('submission_pseudo.csv', index=False)
print("Submission saved to submission_pseudo.csv")
