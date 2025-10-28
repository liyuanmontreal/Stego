# -*- coding: utf-8 -*-
from PIL import Image

def reveal_message(stego_path):
    """Extract a secret message from a stego image.
    从隐写图片中提取隐藏的文字。
    """
    img = Image.open(stego_path).convert('RGB')
    pixels = list(img.getdata())
    bits = []
    for r, g, b in pixels:
        bits.extend([str(r & 1), str(g & 1), str(b & 1)])
    length = int(''.join(bits[:32]), 2)
    msg_bits = bits[32:32 + length]
    chars = [chr(int(''.join(msg_bits[i:i+8]), 2)) for i in range(0, len(msg_bits), 8)]
    message = ''.join(chars)
    print("✅ Extracted message:", message)
    return message
