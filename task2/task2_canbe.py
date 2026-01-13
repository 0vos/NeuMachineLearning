import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import math
from timm.optim.optim_factory import param_groups_layer_decay
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.metrics import f1_score
from PIL import Image
from datetime import datetime
from torch import amp
from contextlib import nullcontext

# Configuration
DATA_DIR = 'NeuMachineLearning-main/neu-plant-seedling-classification-num2-2025/dataset-for-task2'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Hyperparameters
IMG_SIZE = 384
BATCH_SIZE = 12
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 5e-5
EPOCHS = 70
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
LOG_FILE = 'canbe_train.log'
EMA_DECAY = 0.999
MODEL_NAME = 'vit_large_patch16_384.augreg_in21k_ft_in1k'
DROP_PATH_RATE = 0.2
WARMUP_EPOCHS = 5
MIXUP_PROB = 0.4
LAYER_DECAY = 0.8
FREEZE_EPOCHS = 2
WEIGHT_DECAY = 0.05
HF_CACHE_DIR = os.path.join(os.getcwd(), 'hf_cache')
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HF_HUB_CACHE'] = HF_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = HF_CACHE_DIR

def log(message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as log_f:
        log_f.write(line + '\n')

with open(LOG_FILE, 'a', encoding='utf-8') as log_f:
    log_f.write('\n' + '=' * 60 + '\n')
    log_f.write(f"Run started at {datetime.now().isoformat()}\n")

log(f"Using device: {DEVICE}")
log(f"PyTorch Version: {torch.__version__}")
log(f"Torchvision Version: {torch.__version__}")

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
    v2.Resize((448, 448)),
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(15),
    v2.TrivialAugmentWide(), # Automatic augmentation tuned for small datasets
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

# MixUp / CutMix Wrapper
# We need to apply this to a BATCH of data, not single images.
class ModelEMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for key in ema_state.keys():
            ema_state[key].copy_(ema_state[key] * self.decay + model_state[key] * (1.0 - self.decay))

# Dataset
full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)
log(f"Classes: {class_names}")

cutmix = v2.CutMix(num_classes=num_classes)
mixup = v2.MixUp(num_classes=num_classes)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

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

# Model: timm backbone with DropPath
log(f"Initializing timm model: {MODEL_NAME}...")
model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=num_classes,
    drop_path_rate=DROP_PATH_RATE,
)
model = model.to(DEVICE)
ema_helper = ModelEMA(model, decay=EMA_DECAY)
scaler = amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')

if FREEZE_EPOCHS > 0:
    for name, param in model.named_parameters():
        if not name.startswith('head') and 'fc_norm' not in name:
            param.requires_grad = False

param_groups = param_groups_layer_decay(
    model,
    base_lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    layer_decay=LAYER_DECAY,
)
optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training Loop
log("Starting training...")
best_f1 = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    if epoch < WARMUP_EPOCHS:
        current_lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
    else:
        progress = (epoch - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
        current_lr = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Apply MixUp/CutMix
        if torch.rand(1).item() < MIXUP_PROB:
             images, labels = cutmix_or_mixup(images, labels)
        
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema_helper.update(model)
        
        running_loss += loss.item()
        
    # Validation
    eval_model = ema_helper.ema
    eval_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = eval_model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    log(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} Val F1: {val_f1:.4f} LR: {current_lr:.6f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_wts = copy.deepcopy(eval_model.state_dict())
        torch.save(eval_model.state_dict(), 'best_model_final.pth')

log(f"Best Validation F1: {best_f1:.4f}")

# --- Inference ---
log("Generating predictions...")

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

model.load_state_dict(torch.load('best_model_final.pth', map_location=DEVICE))
model.eval()

predictions = []
filenames = []

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(DEVICE)
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

submission_df.to_csv('submission_canbe.csv', index=False)
log("Submission saved to submission_canbe.csv")