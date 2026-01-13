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
from PIL import Image
import torch.nn.functional as F
import gc

# --- Configuration ---
DATA_DIR = 'NeuMachineLearning-main/neu-plant-seedling-classification-num2-2025/dataset-for-task2'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Ensemble Settings
IMG_SIZE_STD = 256 # For DenseNet
IMG_SIZE_HIGH = 384 # For EfficientNet & ResNet
BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

print(f"Using device: {DEVICE}")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True 

set_seed(SEED)

# --- Preprocessing ---
def segment_plant(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(image_np, image_np, mask=mask)
    return result

# --- Datasets ---
# 1. Standard Dataset (For DenseNet - No Masking)
class StandardDataset(Dataset):
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

# 2. Segmented Dataset (For EfficientNet & ResNet - With Masking)
class SegmentedDataset(Dataset):
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
        
        # Apply segmentation
        image_np = np.array(image)
        image_segmented = segment_plant(image_np)
        image = Image.fromarray(image_segmented)
        
        if self.transform:
            image = self.transform(image)
        return image, img_name

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

# --- Transforms ---
# For DenseNet (Standard)
val_transform_std = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE_STD, IMG_SIZE_STD)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# For EfficientNet/ResNet (High Res + Masked)
train_transform_high = v2.Compose([
    v2.ToImage(),
    v2.Resize((400, 400)),
    v2.RandomResizedCrop(IMG_SIZE_HIGH, scale=(0.5, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(45),
    v2.TrivialAugmentWide(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.1),
])

val_transform_high = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE_HIGH, IMG_SIZE_HIGH)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Model Definitions ---
def get_densenet(num_classes):
    model = models.densenet121(weights=None) # Load weights later
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def get_efficientnet(num_classes):
    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

def get_resnet(num_classes):
    print("Initializing ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(DEVICE)

# --- Training Helper for ResNet ---
def train_resnet(num_classes):
    from sklearn.metrics import f1_score  # Ensure f1_score is imported here
    print("\n--- Training ResNet50 for Ensemble ---")
    # Data
    full_dataset = SegmentedImageFolder(root=TRAIN_DIR, transform=train_transform_high)
    
    # 90/10 Split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    indices = torch.randperm(len(full_dataset)).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    train_subset = torch.utils.data.Subset(SegmentedImageFolder(TRAIN_DIR, transform=train_transform_high), train_idx)
    val_subset = torch.utils.data.Subset(SegmentedImageFolder(TRAIN_DIR, transform=val_transform_high), val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = get_resnet(num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_f1 = 0.0
    EPOCHS = 30 # Fast training
    
    cutmix = v2.CutMix(num_classes=num_classes)
    mixup = v2.MixUp(num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
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
        print(f"ResNet Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model_resnet.pth')
            
    print(f"ResNet Training Finished. Best F1: {best_f1:.4f}")
    return 'best_model_resnet.pth'

# --- Inference Helper ---
def predict_model(model, loader, device):
    model.eval()
    probs_list = []
    filenames = []
    
    with torch.no_grad():
        for images, names in loader:
            images = images.to(device)
            
            # TTA: Original + HFlip + VFlip
            out1 = model(images)
            prob1 = torch.softmax(out1, dim=1)
            
            out2 = model(v2.functional.hflip(images))
            prob2 = torch.softmax(out2, dim=1)
            
            out3 = model(v2.functional.vflip(images))
            prob3 = torch.softmax(out3, dim=1)
            
            avg_probs = (prob1 + prob2 + prob3) / 3
            probs_list.extend(avg_probs.cpu().numpy())
            filenames.extend(names)
            
    return filenames, np.array(probs_list)

# --- Main Ensemble Logic ---
if __name__ == "__main__":
    # Get Class Names
    temp_dataset = datasets.ImageFolder(root=TRAIN_DIR)
    class_names = temp_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # 1. Check/Train ResNet
    if not os.path.exists('best_model_resnet.pth'):
        train_resnet(num_classes)
    else:
        print("Found existing ResNet model.")

    # 2. Prepare Test Loaders
    # Loader for DenseNet (Standard)
    test_dataset_std = StandardDataset(root_dir=TEST_DIR, transform=val_transform_std)
    test_loader_std = DataLoader(test_dataset_std, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Loader for EffNet/ResNet (Masked)
    test_dataset_high = SegmentedDataset(root_dir=TEST_DIR, transform=val_transform_high)
    test_loader_high = DataLoader(test_dataset_high, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 3. Inference - Model 1: DenseNet121 (Standard)
    print("\n--- Predicting with DenseNet121 (Standard) ---")
    if os.path.exists('best_model_final.pth'):
        model_d = get_densenet(num_classes).to(DEVICE)
        model_d.load_state_dict(torch.load('best_model_final.pth'))
        names, probs_d = predict_model(model_d, test_loader_std, DEVICE)
        del model_d
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("Warning: best_model_final.pth not found! Skipping DenseNet.")
        probs_d = np.zeros((len(test_dataset_std), num_classes))

    # 4. Inference - Model 2: EfficientNetV2-S (Masked)
    print("\n--- Predicting with EfficientNetV2-S (Masked) ---")
    if os.path.exists('best_model_boost.pth'):
        model_e = get_efficientnet(num_classes).to(DEVICE)
        model_e.load_state_dict(torch.load('best_model_boost.pth'))
        _, probs_e = predict_model(model_e, test_loader_high, DEVICE)
        del model_e
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("Warning: best_model_boost.pth not found! Skipping EfficientNet.")
        probs_e = np.zeros((len(test_dataset_high), num_classes))

    # 5. Inference - Model 3: ResNet50 (Masked)
    print("\n--- Predicting with ResNet50 (Masked) ---")
    if os.path.exists('best_model_resnet.pth'):
        model_r = get_resnet(num_classes).to(DEVICE)
        model_r.load_state_dict(torch.load('best_model_resnet.pth'))
        _, probs_r = predict_model(model_r, test_loader_high, DEVICE)
        del model_r
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("Warning: best_model_resnet.pth not found! Skipping ResNet.")
        probs_r = np.zeros((len(test_dataset_high), num_classes))

    # 6. Weighted Ensemble
    print("\n--- Calculating Ensemble ---")
    # Weights: EfficientNet (Best) > ResNet (New) > DenseNet (Old)
    # Adjust weights based on your confidence in each model
    # EffNet (0.939), DenseNet (0.922), ResNet (Unknown but likely ~0.93)
    
    w_e = 0.5
    w_r = 0.3
    w_d = 0.2
    
    # Normalize weights if models are missing
    if np.sum(probs_e) == 0: w_e = 0
    if np.sum(probs_r) == 0: w_r = 0
    if np.sum(probs_d) == 0: w_d = 0
    
    total_w = w_e + w_r + w_d
    w_e /= total_w
    w_r /= total_w
    w_d /= total_w
    
    print(f"Weights -> EffNet: {w_e:.2f}, ResNet: {w_r:.2f}, DenseNet: {w_d:.2f}")
    
    final_probs = (w_e * probs_e) + (w_r * probs_r) + (w_d * probs_d)
    final_preds = np.argmax(final_probs, axis=1)
    
    predicted_labels = [class_names[p] for p in final_preds]
    
    submission_df = pd.DataFrame({
        'ID': names,
        'Category': predicted_labels
    })
    
    submission_df.to_csv('submission_ensemble_final.csv', index=False)
    print("Submission saved to submission_ensemble_final.csv")
