# Import libraries
import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score
from torchvision import transforms
import NIH_ChestXRay_Dataset_Module as nih

# Import Vision Transformer models
from transformers import ViTForImageClassification, ViTConfig, ViTFeatureExtractor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set paths and hyperparameters
data_dir = "path/to/nih_chest_xray_dataset"  # Update with your path
csv_file = os.path.join(data_dir, "Data_Entry_2017.csv")
batch_size = 32
num_epochs = 10
lr = 1e-4
weight_decay = 1e-5

# Load data
print("Loading data...")
train_loader, val_loader, test_loader, class_weights = nih.get_nih_data_loaders(
    data_dir=data_dir, 
    batch_size=batch_size,
    sample_size=5000,  # Adjust as needed
    test_size=1000,
    balance=True,
    verbose=True
)

# Define ViT model with custom classification head
class ViTForChestXray(torch.nn.Module):
    def __init__(self, num_labels=len(nih.NIHChestXRay.LABELS)):
        super().__init__()
        # Load pretrained ViT
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Replace classification head
        hidden_size = self.vit.config.hidden_size
        self.vit.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

# Create model
print("Initializing Vision Transformer model...")
model = ViTForChestXray()
model.to(device)

# Define optimizer and loss
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.BCEWithLogitsLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training metrics
train_losses = []
val_losses = []
val_aucs = []
best_val_auc = 0.0

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    
    return epoch_loss / batch_count

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    batch_count = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            batch_count += 1
            
            # Store predictions and labels
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
    
    # Calculate metrics
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
    # Calculate AUC for each class and average
    aucs = []
    for i in range(len(nih.NIHChestXRay.LABELS)):
        if np.sum(all_labels[:, i]) > 0:  # Only calculate if there are positive examples
            aucs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
    
    mean_auc = np.mean(aucs)
    return val_loss / batch_count, mean_auc

# Testing function
def test(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Store predictions and labels
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
    
    # Calculate metrics
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
    # Calculate AUC for each class
    class_aucs = {}
    for i, label_name in enumerate(nih.NIHChestXRay.LABELS):
        if np.sum(all_labels[:, i]) > 0:
            class_aucs[label_name] = roc_auc_score(all_labels[:, i], all_preds[:, i])
    
    # Calculate binary accuracy (threshold = 0.5)
    binary_preds = (all_preds > 0.5).astype(float)
    accuracy = np.mean(np.equal(binary_preds, all_labels).astype(float))
    
    return class_aucs, accuracy

# Save checkpoint function
def save_checkpoint(model, epoch, val_auc):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_auc': val_auc
    }
    torch.save(checkpoint, f'vit_chest_xray_epoch_{epoch}_auc_{val_auc:.3f}.pt')

# Training loop
print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    start_time = time.time()
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    
    # Validate
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_aucs.append(val_auc)
    
    # Update learning rate
    scheduler.step()
    
    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        save_checkpoint(model, epoch+1, val_auc)
    
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    print(f"Epoch completed in {epoch_time:.2f} seconds")
    print("-" * 50)

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(val_aucs, label='Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.title('Validation AUC')

plt.tight_layout()
plt.savefig('vit_training_history.png')
plt.show()

# Test the model
print("Evaluating model on test set...")
class_aucs, accuracy = test(model, test_loader, device)

print(f"Test Accuracy: {accuracy:.4f}")
print("Test AUC by class:")
for label, auc in class_aucs.items():
    print(f"  {label}: {auc:.4f}")

# Plot AUCs
plt.figure(figsize=(12, 6))
plt.bar(class_aucs.keys(), class_aucs.values())
plt.xticks(rotation=90)
plt.ylabel('AUC')
plt.title('Test AUC by Class')
plt.tight_layout()
plt.savefig('vit_test_aucs.png')
plt.show()

# Save final model
torch.save(model.state_dict(), 'vit_chest_xray_final.pt')
print("Training completed and model saved!")
