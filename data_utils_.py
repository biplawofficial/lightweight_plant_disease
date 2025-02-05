import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torchvision.datasets import ImageFolder

def split_data(source_dir, dest_dir, train_split=0.8, val_split=0.1):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    folder_images = {}
    
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        if folder_name not in folder_images:
            folder_images[folder_name] = []
        
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                folder_images[folder_name].append(os.path.join(root, file_name))
    
    for folder_name, file_paths in folder_images.items():
        train_dir = os.path.join(dest_dir, 'train', folder_name)
        val_dir = os.path.join(dest_dir, 'val', folder_name)
        test_dir = os.path.join(dest_dir, 'test', folder_name)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        random.shuffle(file_paths)
        
        total_files = len(file_paths)
        train_index = int(total_files * train_split)
        val_index = int(total_files * (train_split + val_split))
        
        train_files = file_paths[:train_index]
        val_files = file_paths[train_index:val_index]
        test_files = file_paths[val_index:]
        
        for file_path in train_files:
            shutil.copy(file_path, os.path.join(train_dir, os.path.basename(file_path)))
        
        for file_path in val_files:
            shutil.copy(file_path, os.path.join(val_dir, os.path.basename(file_path)))
        
        for file_path in test_files:
            shutil.copy(file_path, os.path.join(test_dir, os.path.basename(file_path)))
        
        print(f"Copied {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files for folder {folder_name}")

def display_img(images, labels):
    cols = 8
    rows = (images.shape[0] // cols) + (1 if images.shape[0] % cols != 0 else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
    
    for i in range(images.shape[0]):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        ax.imshow(img)
        ax.axis('off') 
        ax.set_title(f"Label: {labels[i]}")
    
    for j in range(images.shape[0], rows * cols):
        if rows > 1:
            axes[j // cols, j % cols].axis('off')
        else:
            axes[j % cols].axis('off')
    
    plt.tight_layout()
    plt.show()

def confusion_matrix_heatmap(confusion_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

def visualize_dataset(train_dir):
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    data = [{'Class': class_name, 'Count': len(os.listdir(os.path.join(train_dir, class_name)))} for class_name in classes]
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Count', data=df)
    plt.title('Dataset Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 8))
    plt.pie(df['Count'], labels=df['Class'], autopct='%1.1f%%', startangle=90)
    plt.title('Dataset Distribution (Pie Chart)')
    plt.show()
    
    total_images = df['Count'].sum()
    print(f"Total number of images: {total_images}")
    max_class = df.loc[df['Count'].idxmax()]
    min_class = df.loc[df['Count'].idxmin()]
    print(f"Class with maximum images: {max_class['Class']} ({max_class['Count']})")
    print(f"Class with minimum images: {min_class['Class']} ({min_class['Count']})")

