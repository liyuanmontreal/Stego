# Stego 
Image Steganography and AI Detection System.

features:
1. user upload an image , input stego message, generate a stego image.
2. user upload an stego image, reveal stego image.
3. use CNN to detect if this image is an stego image.


Run with:
```bash
pip install -r requirements.txt
streamlit run app/stego_app.py
```


generate stego images
python tools/generate_stego_dataset.py

train model
python detect/train_detector.py