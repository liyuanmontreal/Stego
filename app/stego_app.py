# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from steg.lsb_hide import hide_message
from steg.lsb_reveal import reveal_message
from detect.predict_detector import detect_stego
import tempfile

st.set_page_config(page_title="Stego", page_icon="🕵️‍♀️")
st.title("Image Steganography & AI Detection")

st.header("1️ Hide Message in Image")
orig = st.file_uploader("Upload an image (PNG preferred)", type=["png","jpg"])
msg = st.text_area("Enter your secret message")
if st.button("Generate Stego Image"):
    if orig and msg:
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_in.write(orig.read())
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        hide_message(tmp_in.name, tmp_out.name, msg)
        st.image(tmp_out.name, caption="Generated Stego Image Preview")
        st.download_button("Download Stego Image", open(tmp_out.name, "rb"), "stego.png")

st.header("2 Reveal Message from Image")
stego_img = st.file_uploader("Upload a stego image", type=["png"])
if st.button("Reveal Message"):
    if stego_img:
        tmp_s = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_s.write(stego_img.read())
        message = reveal_message(tmp_s.name)
        st.success(f"Extracted message: {message}")

st.header("3️ AI Detection (Is it a stego image?)")
det_img = st.file_uploader("Upload any image", type=["png","jpg"])
if st.button("Run AI Detection"):
    if det_img:
        tmp_d = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_d.write(det_img.read())
        label, prob = detect_stego(tmp_d.name)
        st.write(f"AI Detection Result: **{label}**")
        st.write(f"Confidence: **{prob:.2f}**")
