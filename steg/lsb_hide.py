# -*- coding: utf-8 -*-
from PIL import Image

def hide_message(in_path, out_path, message):
    """Hide a secret text message inside an image using LSB.
    使用最低有效位(LSB)将文字嵌入图片。
    """
    img = Image.open(in_path).convert('RGB')
    pixels = list(img.getdata())
    bits = ''.join(format(ord(c), '08b') for c in message)
    length = format(len(bits), '032b')
    full_bits = length + bits

    if len(full_bits) > len(pixels) * 3:
        raise ValueError("Message too long to hide in this image. 消息太长，无法嵌入！")

    new_pixels, bit_idx = [], 0
    for r, g, b in pixels:
        if bit_idx < len(full_bits):
            r = (r & ~1) | int(full_bits[bit_idx]); bit_idx += 1
        if bit_idx < len(full_bits):
            g = (g & ~1) | int(full_bits[bit_idx]); bit_idx += 1
        if bit_idx < len(full_bits):
            b = (b & ~1) | int(full_bits[bit_idx]); bit_idx += 1
        new_pixels.append((r, g, b))

    img.putdata(new_pixels)
    img.save(out_path, 'PNG')
    print(f"✅ Hidden message saved to: {out_path}")
