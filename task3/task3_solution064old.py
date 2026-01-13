import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from tqdm import tqdm
import random
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# ==========================================
# Configuration
# ==========================================
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data3/fer_data/fer_data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
SUBMISSION_FILE = os.path.join(SCRIPT_DIR, 'data3/submission.csv')

# Hyperparameters
BATCH_SIZE = 64 * 8  # 8 GPUs
EPOCHS = 40 # Increased epochs for better convergence with mixup
LEARNING_RATE = 1e-4
IMG_SIZE = 64 # 224->64
NUM_FOLDS = 5
SEED = 42
NUM_CLASSES = 6

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Mapping (Folder Name -> Label ID)
CLASS_MAP = {
    'Angry': 0,
    'Fear': 1,
    'Happy': 2,
    'Sad': 3,
    'Surprise': 4,
    'Neutral': 5
}
# Reverse mapping for debugging if needed
IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}

# ==========================================
# Utils & Reproducibility
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ==========================================
# Dataset
# ==========================================
class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        
        if self.is_train:
            # Load training data with labels
            for class_name, label in CLASS_MAP.items():
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_name), label))
        else:
            # Load test data (no labels)
            for img_name in os.listdir(root_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(root_dir, img_name), -1, img_name)) # -1 as dummy label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_train:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        else:
            img_path, _, img_name = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_name

# ==========================================
# Augmentation & Transforms (优化后)
# ==========================================
# 调整为针对 48x48 灰度人脸优化的策略
# 避免不必要的颜色抖动和过度几何变换

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # 随机裁剪 (RandomCrop) 在 FER 任务中非常有效
    transforms.RandomCrop(IMG_SIZE, padding=4), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10), # 减小旋转角度
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)), # 减小几何抖动
    # 只做亮度 (brightness) 和对比度 (contrast) 抖动
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    # 灰度图归一化通常使用 0.5, 0.5
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==========================================
# Mixup Implementation
# ==========================================
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

import timm
# ==========================================
# 3. SOTA-inspired Model: ResNet18 + CBAM Attention
# ==========================================
# 针对小图优化的 ResNet18 架构，加入了 CBAM 注意力机制

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        # 依次应用通道注意力和空间注意力
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        # 引入 CBAM 注意力
        self.cbam = CBAMBlock(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply Attention
        out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet18_FER(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18_FER, self).__init__()
        self.inplanes = 64
        # 针对小图修改的第一层：使用 3x3卷积，步长1 (保留更多细节)，去掉MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 后续的 Residual Block 保持原 ResNet18 结构
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_model():
    # 使用我们定制的 ResNet18_FER
    model = ResNet18_FER(NUM_CLASSES)
    return model

# ==========================================
# Training & Validation Functions
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            if use_mixup and np.random.random() < 0.5: # Apply mixup 50% of the time
                images, targets_a, targets_b, lam = mixup_data(images, labels)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        # Accuracy calculation (only for non-mixup or approximation)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
        
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), correct / total

# ==========================================
# Main Execution
# ==========================================
def main():
    print(f"Using {torch.cuda.device_count()} GPUs!")
    
    # 1. Prepare Data
    full_dataset = FERDataset(TRAIN_DIR, transform=train_transforms, is_train=True)
    targets = [label for _, label in full_dataset.samples]
    
    # 2. K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    fold_results = []
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n{'='*20} Fold {fold+1}/{NUM_FOLDS} {'='*20}")
        
        # Create Subsets
        # Note: We use train_transforms for training subset and val_transforms for validation subset
        # But SubsetRandomSampler just picks indices. We need to handle transforms carefully.
        # A cleaner way is to create two dataset instances.
        
        train_subset = torch.utils.data.Subset(FERDataset(TRAIN_DIR, transform=train_transforms, is_train=True), train_idx)
        val_subset = torch.utils.data.Subset(FERDataset(TRAIN_DIR, transform=val_transforms, is_train=True), val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
        
        # Model Setup
        model = get_model()
        model = model.to(DEVICE)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
        # Loss & Optimizer
        # Label Smoothing helps with overfitting
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scaler = torch.amp.GradScaler('cuda')
        
        best_acc = 0.0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                # Save best model for this fold
                model_save_path = os.path.join(SCRIPT_DIR, f'data3/best_model_fold{fold}.pth')
                torch.save(best_model_state, model_save_path)
        
        print(f"Fold {fold+1} Best Val Acc: {best_acc:.4f}")
        fold_results.append(best_acc)
        best_models.append(model_save_path)

    print(f"\nAverage Val Acc across {NUM_FOLDS} folds: {np.mean(fold_results):.4f}")

    # ==========================================
    # Inference & Submission
    # ==========================================
    print("\nStarting Inference on Test Set...")
    test_dataset = FERDataset(TEST_DIR, transform=val_transforms, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    # Ensemble Predictions
    final_preds = []
    image_names = []
    
    # We will accumulate probabilities from all fold models
    # Also using Test Time Augmentation (Horizontal Flip)
    
    # Prepare models
    loaded_models = []
    for model_path in best_models:
        m = get_model()
        if torch.cuda.device_count() > 1:
            m = nn.DataParallel(m)
        m.load_state_dict(torch.load(model_path))
        m.to(DEVICE)
        m.eval()
        loaded_models.append(m)
        
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Inference"):
            images = images.to(DEVICE)
            
            # TTA: Original + Flip
            images_flipped = torch.flip(images, [3])
            
            batch_probs = torch.zeros(images.size(0), NUM_CLASSES).to(DEVICE)
            
            for model in loaded_models:
                # Original
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                batch_probs += probs
                
                # Flipped
                outputs_flip = model(images_flipped)
                probs_flip = F.softmax(outputs_flip, dim=1)
                batch_probs += probs_flip
            
            # Average over (Models * TTA)
            batch_probs /= (len(loaded_models) * 2)
            
            _, preds = torch.max(batch_probs, 1)
            
            final_preds.extend(preds.cpu().numpy())
            image_names.extend(names)
            
    # Create Submission
    df = pd.DataFrame({
        'image_id': image_names,
        'label': final_preds
    })
    
    # Sort by image_id if needed, but usually just matching filenames is enough
    # Ensure the format matches sample submission if provided
    
    df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")

if __name__ == '__main__':
    main()
