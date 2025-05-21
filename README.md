# Person Re-Identification Project

Simple, easy-to-understand implementation of a person re-identification system using PyTorch and Market-1501 dataset.

---

## 📌 Overview

This project trains a CNN-based model with triplet loss on Market-1501 to learn embeddings that match the same person across different camera views. At inference, it retrieves the top-5 most similar gallery images for each query.

---

## 🔧 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/zaanity/person-Re-id.git
   cd person-Re-id

2. **Create environment & install**
   ```bash
   conda create -n preid python=3.8 -y
   conda activate preid

---

## 🏗 Architecture

1. **Backbone**: ResNet-50 pretrained on ImageNet

2. **Embedding Head**: Linear layer projecting to 128-D, followed by L2 normalization

3. **Loss**: Batch-hard triplet loss (margin = 0.3)


    ```bash
    flowchart LR
    Input --> Preprocess
    Preprocess --> ResNet50
    ResNet50 --> EmbeddingLayer
    EmbeddingLayer --> L2Normalize
    L2Normalize --> Embeddings

##⚙️ Usage

    ```bash
    python train.py \
    --train_dir /path/to/Market-1501/bounding_box_train \
    --query_dir /path/to/Market-1501/query \
    --gallery_dir /path/to/Market-1501/bounding_box_test
