# -*- coding: utf-8 -*-
# ===========================================
# File: train_detector.py
# Function: Train CNN stego detector with validation & plotting
# 功能：带验证集和训练曲线绘制的隐写检测模型训练脚本
# ===========================================

import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from detect.model_cnn import StegoDetector

def train_detector(data_dir='dataset', model_out='stego_detector.pth',
                      batch_size=16, epochs=15, lr=1e-4, val_ratio=0.2):
    """
    Train CNN model with validation and plot training curves.
    使用验证集训练卷积神经网络，并绘制loss和accuracy曲线。
    """

    # --- 1️⃣ 数据集加载 ---
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"✅ Loaded {train_len} training + {val_len} validation samples.")

    # --- 2️⃣ 模型与优化器 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StegoDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, val_accs = [], [], []

    # --- 3️⃣ 训练循环 ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            preds = model(imgs).squeeze()
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # --- 验证阶段 ---
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.float().to(device)
                preds = model(imgs).squeeze()
                loss = criterion(preds, labels)
                val_loss += loss.item() * imgs.size(0)
                correct += ((preds > 0.5).int() == labels.int()).sum().item()
                total += imgs.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}")

    # --- 4️⃣ 保存模型 ---
    torch.save(model.state_dict(), model_out)
    print(f"🎯 Model saved to {model_out}")

    # --- 5️⃣ 绘制曲线 ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(val_accs, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_detector(data_dir='dataset', epochs=15)
