import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import ViTForImageClassification

class ViTBinaryClassifier(torch.nn.Module):
    """
    Binary classifier using Vision Transformer to detect if there's any finding or not
    """
    def __init__(self):
        super().__init__()
        # Load pretrained ViT for binary classification
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=1,  # Just one output for binary
            ignore_mismatched_sizes=True
        )
        
        # Replace classification head
        hidden_size = self.vit.config.hidden_size
        self.vit.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, 1)
        )
        
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits


class ViTDiseaseClassifier(torch.nn.Module):
    """
    Multi-label classifier using Vision Transformer to detect specific disease findings
    """
    def __init__(self, num_labels=14, labels=None):  # Excluding "No Finding"
        super().__init__()
        self.labels = labels
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


def train_binary_model(model, dataloader, criterion, optimizer, device, epoch=0):
    """
    Train the binary classifier for one epoch
    """
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        
        # Convert multi-label to binary (1 if any disease, 0 if "No Finding")
        binary_labels = (labels.sum(dim=1) > 0).float().unsqueeze(1).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, binary_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    
    return epoch_loss / batch_count


def train_multilabel_model(model, dataloader, criterion, optimizer, device, epoch=0):
    """
    Train the multi-label disease classifier for one epoch
    """
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    # Get number of outputs from model
    out_features = model.vit.classifier[-1].out_features
    print(f"Model output features: {out_features}")
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Skip completely negative examples (no diseases)
        if (labels.sum(dim=1) == 0).all():
            continue
        
        # Always exclude "No Finding" column (first column) if dataset includes it
        disease_labels = labels[:, 1:] if labels.shape[1] == 15 else labels
        
        # Debug info on first batch
        if batch_idx == 0:
            print(f"Input label shape: {labels.shape}, Disease labels shape: {disease_labels.shape}")
        
        # Verify disease_labels matches model output size
        if disease_labels.shape[1] != out_features:
            print(f"WARNING: Shape mismatch! Labels: {disease_labels.shape[1]}, Model outputs: {out_features}")
            # Force correct shape by selecting only needed columns
            if disease_labels.shape[1] > out_features:
                disease_labels = disease_labels[:, :out_features]
            else:
                # This case shouldn't happen if model is correctly initialized with 14 outputs
                raise ValueError(f"Model has {out_features} outputs but labels only have {disease_labels.shape[1]} disease classes")
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, disease_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    
    if batch_count == 0:
        return 0  # Avoid division by zero
    return epoch_loss / batch_count


def validate_binary_model(model, dataloader, criterion, device):
    """
    Validate the binary classifier
    """
    model.eval()
    val_loss = 0
    batch_count = 0
    all_binary_labels = []
    all_binary_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Convert multi-label to binary
            binary_labels = (labels.sum(dim=1) > 0).float().unsqueeze(1).to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, binary_labels)
            
            val_loss += loss.item()
            batch_count += 1
            
            # Store predictions and labels
            all_binary_labels.append(binary_labels.cpu().numpy())
            all_binary_preds.append(torch.sigmoid(outputs).cpu().numpy())
    
    # Calculate metrics
    all_binary_labels = np.vstack(all_binary_labels)
    all_binary_preds = np.vstack(all_binary_preds)
    
    # Calculate AUC and accuracy
    binary_auc = roc_auc_score(all_binary_labels, all_binary_preds)
    binary_acc = accuracy_score(all_binary_labels, (all_binary_preds > 0.5))
    
    return val_loss / batch_count, binary_auc, binary_acc


def validate_multilabel_model(model, dataloader, criterion, device, disease_labels):
    """
    Validate the multi-label disease classifier
    """
    model.eval()
    val_loss = 0
    batch_count = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Skip examples with no diseases
            if (labels.sum(dim=1) == 0).all():
                continue
            
            # Exclude the "No Finding" column if we have 15 labels (class 0 is typically "No Finding")
            disease_labels_batch = labels[:, 1:] if labels.shape[1] == 15 else labels
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, disease_labels_batch)
            
            val_loss += loss.item()
            batch_count += 1
            
            # Store predictions and labels
            all_labels.append(disease_labels_batch.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
    
    if batch_count == 0:
        return 0, 0, {}  # No valid batches
    
    # Calculate metrics
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
    # Calculate AUC for each class and average
    aucs = []
    class_aucs = {}
    
    for i, label_name in enumerate(disease_labels):
        if np.sum(all_labels[:, i]) > 0:  # Only calculate if there are positive examples
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)
            class_aucs[label_name] = auc
    
    mean_auc = np.mean(aucs) if aucs else 0
    return val_loss / batch_count, mean_auc, class_aucs


def hierarchical_inference(binary_model, disease_model, images, device, disease_labels):
    """
    Perform hierarchical inference using both models
    First determine if there's any finding using the binary model,
    then classify specific diseases only for those with findings
    """
    binary_model.eval()
    disease_model.eval()
    
    with torch.no_grad():
        # First determine if there's any finding
        binary_output = torch.sigmoid(binary_model(images))
        
        # Use len(disease_labels) instead of getting from the model
        num_outputs = len(disease_labels)
        
        # For images with findings (above threshold), run disease classifier
        disease_predictions = torch.zeros((images.size(0), num_outputs), device=device)
        finding_mask = binary_output.squeeze() > 0.5
        
        if finding_mask.sum() > 0:
            # Only process images predicted to have findings
            disease_output = torch.sigmoid(disease_model(images[finding_mask]))
            disease_predictions[finding_mask] = disease_output
        
    return binary_output, disease_predictions


def test_hierarchical_model(binary_model, disease_model, dataloader, device, disease_labels):
    """
    Test the hierarchical model (binary + disease classifier)
    """
    binary_model.eval()
    disease_model.eval()
    
    all_binary_labels = []
    all_binary_preds = []
    all_disease_labels = []
    all_disease_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Convert labels to binary (any finding vs no finding)
            binary_labels = (labels.sum(dim=1) > 0).float().unsqueeze(1)
            
            # Exclude "No Finding" column for disease labels if present
            disease_labels_batch = labels[:, 1:] if labels.shape[1] == 15 else labels
            
            # Perform hierarchical inference
            binary_preds, disease_preds = hierarchical_inference(
                binary_model, disease_model, images, device, disease_labels
            )
            
            # Store predictions and labels
            all_binary_labels.append(binary_labels.numpy())
            all_binary_preds.append(binary_preds.cpu().numpy())
            all_disease_labels.append(disease_labels_batch.cpu().numpy())
            all_disease_preds.append(disease_preds.cpu().numpy())
    
    # Concatenate results
    all_binary_labels = np.vstack(all_binary_labels)
    all_binary_preds = np.vstack(all_binary_preds)
    all_disease_labels = np.vstack(all_disease_labels)
    all_disease_preds = np.vstack(all_disease_preds)
    
    # Calculate binary metrics
    binary_auc = roc_auc_score(all_binary_labels, all_binary_preds)
    binary_acc = accuracy_score(all_binary_labels, (all_binary_preds > 0.5))
    
    # Calculate disease metrics
    disease_aucs = {}
    for i, label_name in enumerate(disease_labels):
        if np.sum(all_disease_labels[:, i]) > 0:
            disease_aucs[label_name] = roc_auc_score(all_disease_labels[:, i], all_disease_preds[:, i])
    
    mean_disease_auc = np.mean(list(disease_aucs.values())) if disease_aucs else 0
    
    return {
        'binary_auc': binary_auc,
        'binary_accuracy': binary_acc,
        'disease_aucs': disease_aucs,
        'mean_disease_auc': mean_disease_auc
    }


def visualize_hierarchical_predictions(binary_model, disease_model, dataloader, device, disease_labels, num_examples=4):
    """
    Visualize predictions from the hierarchical model
    """
    binary_model.eval()
    disease_model.eval()
    
    # Get a batch of samples
    images, labels = next(iter(dataloader))
    
    # Exclude "No Finding" column for disease labels if present
    disease_labels_batch = labels[:, 1:] if labels.shape[1] == 15 else labels
    
    # Perform hierarchical inference
    binary_preds, disease_preds = hierarchical_inference(
        binary_model, disease_model, images.to(device), device, disease_labels
    )
    
    binary_preds = binary_preds.cpu().numpy()
    disease_preds = disease_preds.cpu().numpy()
    
    # Convert to binary predictions
    binary_pred_labels = (binary_preds > 0.5).astype(int)
    disease_pred_labels = (disease_preds > 0.5).astype(int)
    
    # Plot images with their labels
    fig, axes = plt.subplots(num_examples, 1, figsize=(15, 5*num_examples))
    if num_examples == 1:
        axes = [axes]
    
    for i in range(min(num_examples, len(images))):
        # Display the image
        img = images[i].permute(1, 2, 0).numpy()
        # Normalize for visualization
        img = (img - img.min()) / (img.max() - img.min())
        axes[i].imshow(img, cmap='gray')
        
        # Get actual binary label
        actual_binary = 1 if labels[i].sum() > 0 else 0
        
        # Create label text - adjust indices based on disease_labels_batch for actual labels
        actual_labels = []
        for j, label in enumerate(disease_labels):
            # If we have 15 labels, then we need to check j+1 in the original labels 
            # since disease_labels excludes "No Finding"
            idx = j+1 if labels.shape[1] == 15 else j
            if idx < labels.size(1) and labels[i][idx] == 1:
                actual_labels.append(label)
                
        predicted_labels = [disease_labels[j] for j in range(len(disease_labels)) if disease_pred_labels[i][j] == 1]
        
        # If no positive labels found, show "No Finding"
        if len(actual_labels) == 0:
            actual_labels = ["No Finding"]
        if len(predicted_labels) == 0 or binary_pred_labels[i][0] == 0:
            predicted_labels = ["No Finding"]
        
        # Display confidence scores
        conf_text = f"Binary confidence: {binary_preds[i][0]:.2f}\n"
        if binary_pred_labels[i][0] == 1:  # Only show disease confidences if binary prediction is positive
            conf_text += "\n".join([f"{label}: {disease_preds[i][j]:.2f}"
                                for j, label in enumerate(disease_labels)
                                if disease_preds[i][j] > 0.3])
        
        # Set title and text
        axes[i].set_title(f"Example {i+1}")
        axes[i].set_xlabel(
            f"Actual binary: {'Finding' if actual_binary else 'No Finding'} | "
            f"Predicted binary: {'Finding' if binary_pred_labels[i][0] else 'No Finding'}\n\n"
            f"Actual diseases: {', '.join(actual_labels)}\n"
            f"Predicted diseases: {', '.join(predicted_labels)}\n\n"
            f"Confidence Scores:\n{conf_text}"
        )
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.show() 