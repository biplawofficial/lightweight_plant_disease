# Lightweight Plant Disease Classification

## Overview

This project is a deep learning-based image classification system designed to identify plant diseases from images. It leverages lightweight architectures, specifically **MNASNet1.0**, making it suitable for deployment on mobile or edge devices with limited computational resources. The project includes utilities for data processing and visualization, as well as Jupyter notebooks for training both a base model and an enhanced version with Squeeze-and-Excitation (SE) blocks.

## Key Features

- **Efficient Architectures**: Utilizes MNASNet for low-latency inference on varying hardware.
- **Enhanced Feature Extraction**: Includes an implementation with **Squeeze-and-Excitation (SE) Blocks** to improved model accuracy by adaptively recalibrating channel-wise feature responses.
- **Comprehensive Data Tools**: `data_utils_.py` provides functions for automatic dataset splitting (Train/Val/Test) and detailed visualization (Confusion Matrices, Class Distribution).
- **Custom Training Loop**: Tracks critical metrics like Loss, Accuracy, Precision, Recall, and F1 Score throughout the training process.
- **Optimized Training**: Uses the `AdamW` optimizer and `StepLR` scheduler for effective learning rate management.

## File Structure

### 1. `data_utils_.py`

A utility script containing core functions for data management:

- **Dataset Splitting**: Automatically splits raw image folders into `train` (80%), `val` (10%), and `test` (10%) directories.
- **Visualization**: Functions to display image grids with labels (`display_img`) and plot confusion matrices (`confusion_matrix_heatmap`).
- **Analysis**: Generates bar charts and pie charts to visualize class distribution and identify imbalances (`visualize_dataset`).

### 2. `train.ipynb` (Base Model)

The primary training notebook for the standard MNASNet model:

- Implements a custom classifier layer for **61 plant disease classes**.
- Defines the training loop with performance metric tracking.
- Handles data preprocessing and augmentation.
- Saves the best-performing model as `mnasnet.pth`.

### 3. `train1_se.ipynb` (Enhanced Model)

An advanced training notebook incorporating **SE Blocks**:

- Modifies the MNASNet architecture to include Squeeze-and-Excitation blocks.
- Follows the same rigorous training and evaluation pipeline as the base model.
- Saves the enhanced model as `seblock.pth`.

## Getting Started

### Prerequisites

To run this project, you need Python installed along with the following libraries:

```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn tqdm jupyter
```

### Installation & Usage

1.  **Clone the Repository**:

    ```bash
    git clone <repository-url>
    cd lightweight_plant_disease
    ```

2.  **Prepare the Dataset**:
    Ensure your dataset is organized correctly. The code expects a specific directory structure.
    ![Dataset Directory Structure](gitpng.drawio.png)

3.  **Run the Training**:
    - Launch the Jupyter Notebook server:
      ```bash
      jupyter notebook
      ```
    - Open **`train.ipynb`** (for the base model) or **`train1_se.ipynb`** (for the SE model).
    - **Crucial Step**: Locate the `dataset_dir` or `data_dir` variable in the first few cells of the notebook. Update the path string to match the actual location of your dataset on your local storage.
    - Click **"Cell"** > **"Run All"** to execute the entire training pipeline.

## Contact

For any help or inquiries, please contact **biplawofficial@gmail.com**.
