import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import random
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# ==========================================
# Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data3/fer_data/fer_data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
SUBMISSION_FILE = os.path.join(SCRIPT_DIR, 'data3/submission.csv')

# Hyperparameters
BATCH_SIZE = 64 * 8  # 8 GPUs
EPOCHS = 60 # Reduced epochs per model as we train 3 models
LEARNING_RATE = 1e-4
SEED = 42
NUM_CLASSES = 6

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Mapping
CLASS_MAP = {
    'Angry': 0,
    'Fear': 1,
    'Happy': 2,
    'Sad': 3,
    'Surprise': 4,
    'Neutral': 5
}

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
            for class_name, label in CLASS_MAP.items():
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_name), label))
        else:
            for img_name in os.listdir(root_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(root_dir, img_name), -1, img_name))

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
# Augmentation & Transforms
# ==========================================
def get_transforms(img_size):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=4 if img_size <= 64 else 16),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, val_transforms

# ==========================================
# Mixup
# ==========================================
def mixup_data(x, y, alpha=1.0, use_cuda=True):
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

# ==========================================
# Models
# ==========================================

# --- Model 1: ResNet18 + CBAM (48x48) ---
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
        self.cbam = CBAMBlock(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
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

# --- Model 2: ResNet50 (224x224) ---
def get_resnet50():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# --- Model 3: DenseNet121 (224x224) ---
def get_densenet121():
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    return model

# ==========================================
# Training
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
            if use_mixup and np.random.random() < 0.5:
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

def train_model(model_name, model, train_loader, val_loader, class_weights):
    print(f"\n{'='*20} Training {model_name} {'='*20}")
    model = model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0
    save_path = os.path.join(SCRIPT_DIR, f'data3/best_model_{model_name}.pth')
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best {model_name} (Acc: {best_acc:.4f})")
            
    print(f"Best Val Acc for {model_name}: {best_acc:.4f}")
    return save_path

def main():
    print(f"Using {torch.cuda.device_count()} GPUs!")
    
    # 1. Prepare Data Splits (Indices)
    # We need consistent splits across models
    full_dataset_dummy = FERDataset(TRAIN_DIR, transform=None, is_train=True)
    train_size = int(0.8 * len(full_dataset_dummy))
    indices = list(range(len(full_dataset_dummy)))
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    # Class Weights
    class_counts = [3963, 4097, 7192, 4862, 3202, 4959]
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(class_counts) * c) for c in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    
    # 2. Define Models and their Configs
    models_config = [
        {
            'name': 'resnet18_fer',
            'model_fn': lambda: ResNet18_FER(NUM_CLASSES),
            'img_size': 48
        },
        {
            'name': 'resnet50',
            'model_fn': get_resnet50,
            'img_size': 224
        },
        {
            'name': 'densenet121',
            'model_fn': get_densenet121,
            'img_size': 224
        }
    ]
    
    trained_model_paths = []
    
    # 3. Train Loop
    for config in models_config:
        img_size = config['img_size']
        train_tf, val_tf = get_transforms(img_size)
        
        # Create Datasets with specific transforms
        train_subset = torch.utils.data.Subset(FERDataset(TRAIN_DIR, transform=train_tf, is_train=True), train_idx)
        val_subset = torch.utils.data.Subset(FERDataset(TRAIN_DIR, transform=val_tf, is_train=True), val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
        
        model = config['model_fn']()
        path = train_model(config['name'], model, train_loader, val_loader, class_weights_tensor)
        trained_model_paths.append({'name': config['name'], 'path': path, 'img_size': img_size, 'model_fn': config['model_fn']})

    # 4. Ensemble Inference
    print("\nStarting Ensemble Inference...")
    
    # Load all models
    loaded_models = []
    for info in trained_model_paths:
        m = info['model_fn']()
        if torch.cuda.device_count() > 1:
            m = nn.DataParallel(m)
        m.load_state_dict(torch.load(info['path']))
        m.to(DEVICE)
        m.eval()
        loaded_models.append({'model': m, 'img_size': info['img_size']})
        
    # We need to iterate over the test set. 
    # Since models need different input sizes, we can't use a single DataLoader easily if we want to batch.
    # BUT, we can use the largest size (224) and resize down for the 48 model? 
    # Or just create multiple DataLoaders and iterate them in zip?
    # Zip is risky if shuffle is True, but for Test set shuffle is False.
    
    # Create DataLoaders for each size
    test_loaders = {}
    unique_sizes = set(m['img_size'] for m in loaded_models)
    
    # We need the image names, so we need to access the dataset directly or return them.
    # Let's use the first loader for names.
    
    loaders_list = []
    for size in unique_sizes:
        _, val_tf = get_transforms(size)
        ds = FERDataset(TEST_DIR, transform=val_tf, is_train=False)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        test_loaders[size] = dl
        
    # We assume all loaders return data in same order (shuffle=False)
    # We will iterate the first loader to control the loop, and fetch from others?
    # Actually, since batch size is same, we can zip the iterators.
    
    # Let's simplify: Just run inference for each model separately and store probabilities.
    # Then average probabilities.
    
    all_probs = None
    image_names = []
    
    for i, info in enumerate(trained_model_paths):
        print(f"Inference for {info['name']}...")
        size = info['img_size']
        loader = test_loaders[size]
        
        model_probs = []
        current_names = []
        
        model = loaded_models[i]['model']
        
        with torch.no_grad():
            for images, names in tqdm(loader, desc=f"Infer {info['name']}"):
                images = images.to(DEVICE)
                # TTA
                images_flip = torch.flip(images, [3])
                
                out1 = model(images)
                out2 = model(images_flip)
                
                probs = (F.softmax(out1, dim=1) + F.softmax(out2, dim=1)) / 2
                model_probs.append(probs.cpu().numpy())
                
                if i == 0: # Only collect names once
                    current_names.extend(names)
        
        model_probs = np.concatenate(model_probs, axis=0)
        
        if all_probs is None:
            all_probs = model_probs
            image_names = current_names
        else:
            all_probs += model_probs
            
    # Average
    avg_probs = all_probs / len(trained_model_paths)
    final_preds = np.argmax(avg_probs, axis=1)
    
    df = pd.DataFrame({'ID': image_names, 'Emotion': final_preds})
    df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")

if __name__ == '__main__':
    main()
