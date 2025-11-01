# -*- coding: utf-8 -*-
# ===========================================
# File: generate_stego_dataset.py
# Function: Generate stego images for training dataset
# 功能：从原图批量生成隐写图片数据集，用于AI检测模型训练
# ===========================================

import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import string
from steg.lsb_hide import hide_message
from PIL import Image

def random_message(length=30):
    """Generate a random alphanumeric string."""
    chars = string.ascii_letters + string.digits + " .,!?;:"
    return ''.join(random.choice(chars) for _ in range(length))

def generate_stego_dataset(clean_dir='dataset/clean', stego_dir='dataset/stego'):
    """
    Generate stego images by hiding random messages inside clean images.
    从clean图片中生成隐写图片。
    :param clean_dir: Directory with original clean images
    :param stego_dir: Output directory for generated stego images
    """
    os.makedirs(stego_dir, exist_ok=True)

    clean_images = [f for f in os.listdir(clean_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(clean_images)} clean images in {clean_dir}")

    count = 0
    for img_name in clean_images:
        in_path = os.path.join(clean_dir, img_name)
        out_path = os.path.join(stego_dir, os.path.splitext(img_name)[0] + "_stego.png")

        try:
            msg = random_message(random.randint(20, 100))
            hide_message(in_path, out_path, msg)
            count += 1
        except Exception as e:
            print(f"❌ Error with {img_name}: {e}")

    print(f" Successfully generated {count} stego images in {stego_dir}")

if __name__ == "__main__":
    generate_stego_dataset()
