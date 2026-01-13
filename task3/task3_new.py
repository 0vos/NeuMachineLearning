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
EPOCHS = 80 # Increased epochs for better convergence with mixup
LEARNING_RATE = 1e-4
IMG_SIZE = 224 # 224->64
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
    transforms.Resize((224, 224)), # 必须是 224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 灰度图只做这两个
    transforms.ToTensor(),
    # 使用 ImageNet 的均值和方差，因为我们用了预训练权重
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

import torchvision.models as models

# ==========================================
# Model: Pretrained ResNet18
# ==========================================
class FaceResNet18(nn.Module):
    def __init__(self, num_classes=6, freeze_layers=True):
        super(FaceResNet18, self).__init__()
        # 1. 加载 ImageNet 预训练模型
        # ... (加载逻辑保持不变)
        try:
            from torchvision.models import ResNet18_Weights
            self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except ImportError:
            self.base_model = models.resnet18(pretrained=True)
            
        # 2. 【新增优化】冻结大部分基础特征提取层
        if freeze_layers:
            for name, param in self.base_model.named_parameters():
                # 冻结所有层，除了最后的全连接层（fc）和 BN 层（BatchNorm）
                # BN 层不冻结（或只冻结运行统计信息）通常能获得更好的效果
                if 'fc' not in name and 'bn' not in name:
                    param.requires_grad = False
        
        # 3. 修改全连接层 (fc)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.3), 
            nn.Linear(num_ftrs, num_classes)
        )
    def forward(self, x):
        return self.base_model(x)

def get_model():
    model = FaceResNet18(NUM_CLASSES)
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


from sklearn.model_selection import train_test_split
# ==========================================
# Main Execution (无 K-Fold 版本)
# ==========================================
def main():
    print(f"Using {torch.cuda.device_count()} GPU(s)!")
    
    # 1. 准备所有数据的索引和标签
    # 这里我们只是为了获取总长度和标签来进行切分
    # 真正的加载在后面
    dummy_dataset = FERDataset(TRAIN_DIR, transform=None, is_train=True)
    targets = [label for _, label in dummy_dataset.samples]
    indices = list(range(len(targets)))
    
    # 2. 划分训练集和验证集 (80% 训练, 20% 验证)
    # stratify=targets 保证切分后各类别的比例与原始数据一致（很重要！）
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=SEED, 
        shuffle=True, 
        stratify=targets
    )
    
    print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    # 3. 创建数据集对象
    # 关键点：我们要创建两个独立的数据集对象，分别绑定不同的 Transform
    train_dataset_full = FERDataset(TRAIN_DIR, transform=train_transforms, is_train=True)
    val_dataset_full = FERDataset(TRAIN_DIR, transform=val_transforms, is_train=True)
    
    # 使用 Subset 指定索引
    train_subset = torch.utils.data.Subset(train_dataset_full, train_idx)
    val_subset = torch.utils.data.Subset(val_dataset_full, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # 4. 初始化模型 (单次)
    model = get_model()
    model = model.to(DEVICE)
    # 因为你已经在外部设置了 CUDA_VISIBLE_DEVICES，这里不需要 DataParallel，或者保持原样也没事
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0
    
    # 5. 开始训练 (只跑这一个循环)
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, 'data3/best_model.pth'))
            print(f"Saved Best Model with Acc: {best_acc:.4f}")
            
    print(f"\nTraining Finished. Best Val Acc: {best_acc:.4f}")

    # ==========================================
    # Inference (单模型推理)
    # ==========================================
    print("\nStarting Inference on Test Set...")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'data3/best_model.pth')))
    model.eval()
    
    test_dataset = FERDataset(TEST_DIR, transform=val_transforms, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    final_preds = []
    image_names = []
    
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Inference"):
            images = images.to(DEVICE)
            
            # TTA: Original + Flip
            images_flipped = torch.flip(images, [3])
            
            # Forward pass
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            outputs_flip = model(images_flipped)
            probs_flip = F.softmax(outputs_flip, dim=1)
            
            # Average probabilities
            batch_probs = (probs + probs_flip) / 2
            
            _, preds = torch.max(batch_probs, 1)
            
            final_preds.extend(preds.cpu().numpy())
            image_names.extend(names)
            
    # Submission
    df = pd.DataFrame({'ID': image_names, 'Emotion': final_preds})
    df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")

if __name__ == '__main__':
    main()
    