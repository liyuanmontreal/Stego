# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from PIL import Image
from detect.model_cnn import StegoDetector

def detect_stego(image_path, model_path='stego_detector.pth'):
    model = StegoDetector()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        prob = model(tensor).item()
    label = "Stego image" if prob > 0.5 else "Clean image"
    print(f"AI detection result: {label} (confidence {prob:.2f})")
    return label, prob
