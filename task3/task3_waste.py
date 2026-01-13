import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models, datasets
from PIL import Image
from tqdm import tqdm
import numpy as np

# Configuration
DATA_DIR = 'data3/fer_data/fer_data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
SUBMISSION_FILE = 'data3/submission.csv'
BATCH_SIZE = 128 * 8  # Adjust based on GPU memory, 8 GPUs available
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
IMG_SIZE = 224 # ResNet standard input size

# Class Mapping
# 0: Angry, 1: Fear, 2: Happy, 3: Sad, 4: Surprise, 5: Neutral
# Folder names: Angry, Fear, Happy, Neutral, Sad, Surprise
# ImageFolder sorts classes alphabetically: Angry, Fear, Happy, Neutral, Sad, Surprise
# Let's verify the mapping matches the requirement.
# Alphabetical:
# Angry -> 0
# Fear -> 1
# Happy -> 2
# Neutral -> 3  <-- This is different from user requirement (Neutral is 5)
# Sad -> 4      <-- This is different from user requirement (Sad is 3)
# Surprise -> 5 <-- This is different from user requirement (Surprise is 4)

# User Requirement:
# 0: Angry
# 1: Fear
# 2: Happy
# 3: Sad
# 4: Surprise
# 5: Neutral

# We need to handle this mapping manually.

class_to_idx_user = {
    'Angry': 0,
    'Fear': 1,
    'Happy': 2,
    'Sad': 3,
    'Surprise': 4,
    'Neutral': 5
}

# Custom Dataset to handle specific class mapping
class CustomImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        # Override class_to_idx with user requirement
        class_to_idx = {cls_name: class_to_idx_user[cls_name] for cls_name in classes}
        return classes, class_to_idx

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
        image = Image.open(img_path).convert('RGB') # Convert to RGB for ResNet
        if self.transform:
            image = self.transform(image)
        return image, img_name

def main():
    print(f"Using device: {DEVICE}")
    print(f"GPU Count: {torch.cuda.device_count()}")

    # Data Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    full_train_dataset = CustomImageFolder(root=TRAIN_DIR, transform=train_transform)
    
    # Split into Train and Val
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Apply val_transform to val_dataset (Need to be careful as random_split doesn't change transform)
    # A cleaner way is to create two datasets and use indices, but for simplicity, we'll use the same transform 
    # or just accept augmentation on validation (suboptimal) or override dataset.transform.
    # Let's do it properly by subsetting.
    
    # Re-instantiate for validation to apply correct transform
    val_dataset_full = CustomImageFolder(root=TRAIN_DIR, transform=val_transform)
    # Use the same indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {full_train_dataset.class_to_idx}")

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    model = model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/((pbar.n)+1), 'acc': 100 * correct / total})
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_task3.pth')
            print("Saved best model.")

    print("Training complete.")

    # Inference
    print("Starting Inference...")
    test_dataset = TestDataset(root_dir=TEST_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # Load best model
    # If DataParallel was used, the state_dict keys have 'module.' prefix.
    # We need to handle loading carefully if we are not wrapping in DataParallel again or if we are.
    # Since we are in the same script run, 'model' is already wrapped. 
    # But to be safe and consistent with "Load best model", let's reload.
    
    model = models.resnet50(weights=None) # Structure only
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.load_state_dict(torch.load('best_model_task3.pth'))
    model.eval()
    
    predictions = []
    ids = []
    
    with torch.no_grad():
        for inputs, img_names in tqdm(test_loader, desc="Inference"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.extend(predicted.cpu().numpy())
            ids.extend(img_names)
            
    # Create Submission
    df = pd.DataFrame({
        'ID': ids,
        'Emotion': predictions
    })
    
    # Ensure sorting or order if necessary, but usually ID matching is enough.
    # The sample submission had IDs. Let's just save what we have.
    df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")

if __name__ == '__main__':
    main()
