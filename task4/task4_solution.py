import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Configuration
TRAIN_DIR = '/home/algo/video_agent_group/qianqian/NeuMachineLearning-main/task4/detection/train'
TEST_DIR = '/home/algo/video_agent_group/qianqian/NeuMachineLearning-main/task4/detection/test'
TRAIN_CSV = '/home/algo/video_agent_group/qianqian/NeuMachineLearning-main/task4/detection/fovea_localization_train_GT.csv'
IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FoveaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, img_size=IMG_SIZE):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = int(row['data'])
        img_name = f"{img_id:04d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        # Resize image
        image = image.resize((self.img_size, self.img_size))
        
        if self.transform:
            image = self.transform(image)
            
        # Scale coordinates to [0, 1]
        x = row['Fovea_X']
        y = row['Fovea_Y']
        
        x_norm = x / w
        y_norm = y / h
        
        label = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        
        return image, label

class FoveaTestDataset(Dataset):
    def __init__(self, img_dir, transform=None, img_size=IMG_SIZE):
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        image_resized = image.resize((self.img_size, self.img_size))
        
        if self.transform:
            image_tensor = self.transform(image_resized)
        else:
            image_tensor = transforms.ToTensor()(image_resized)
            
        return image_tensor, img_name, w, h

def get_model():
    model = models.resnet50(pretrained=True)
    # Modify the last layer for regression (2 outputs: x, y)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, 2),
        nn.Sigmoid() # Output in [0, 1]
    )
    return model

def train():
    # Transforms
    train_transform = transforms.Compose([
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset
    full_dataset = FoveaDataset(TRAIN_CSV, TRAIN_DIR, transform=train_transform)
    
    # Split train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = get_model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        
        val_loss = val_loss / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_task4.pth')
            print("Saved best model")

def predict():
    print("Starting prediction...")
    model = get_model()
    model.load_state_dict(torch.load('best_model_task4.pth'))
    model = model.to(DEVICE)
    model.eval()
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = FoveaTestDataset(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    results = []
    
    with torch.no_grad():
        for image, img_name, w, h in test_loader:
            image = image.to(DEVICE)
            output = model(image)
            output = output.cpu().numpy()[0]
            
            # Scale back
            pred_x = output[0] * w.item()
            pred_y = output[1] * h.item()
            
            # Extract ID from filename (e.g., 0081.jpg -> 81)
            img_id = int(os.path.splitext(img_name[0])[0])
            
            results.append({'ImageID': f"{img_id}_Fovea_X", 'value': pred_x})
            results.append({'ImageID': f"{img_id}_Fovea_Y", 'value': pred_y})
            
    df = pd.DataFrame(results)
    df.to_csv('submission_task4.csv', index=False)
    print("Submission saved to submission_task4.csv")

if __name__ == '__main__':
    train()
    predict()
