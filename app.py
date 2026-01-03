import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import base64
import time
result_placeholder = st.empty()
@st.cache_data
def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
#############################################################
#Decoding pictures in the App UI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_local_image(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_img = "https://s.abcnews.com/images/Health/masks-tokyo-gty-ps-230314_1678802239571_hpMain_16x9_1600.jpg"
arrow_gif = load_local_image("arrow.gif")
footer1_gif = load_local_image("student.gif")
footer2_gif = load_local_image("teacher.gif")
footer3_gif = load_local_image("python.gif")
#Main streamlit title
st.set_page_config(page_title="Mask Detection")
#############################################################
#Background
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #0f2027,
        #203a43,
        #2c5364
    );
}
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)
#############################################################
#trained model function to be called within "get answer"
@st.cache_resource
def load_model():
    return YOLO("best.pt")
#############################################################
#Import some Fonts that will be used in streamlit UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Dancing+Script:wght@500;600&display=swap');
body, h1, h2, h3, p {
    font-family: 'Poppins', sans-serif;
}
</style>
""", unsafe_allow_html=True)
#############################################################
#Card style
#Is a way of Adding a more "premium" or "professional" aesthetic that goes beyond Streamlit's default, basic styling.
st.markdown("""
<style>
[data-testid="stMainBlockContainer"] {
    max-width: 800px;
    padding-top: 2rem;
    margin: auto;
}
.card {
    background-color: rgba(255,255,255,0.95);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)
#############################################################
#Buttons style
st.markdown("""
<style>
/* ---- إعدادات عامة للزر ---- */
div.stButton > button {
    position: relative;
    white-space: nowrap;
    padding: 0.7rem 3.2rem 0.7rem 2rem; 
    font-size: 15px;
    font-weight: 600;
    border-radius: 14px;
    transition: all 0.35s ease;
    min-width: 200px;
    cursor: pointer;
}
/* نص الزر */
div.stButton > button > div {
    transition: transform 0.35s ease;
    z-index: 1;
    position: relative;
}
/* الأيقونة */
div.stButton > button::after {
    position: absolute;
    right: 1.2rem;
    opacity: 0;
    transform: translateX(12px);
    transition: all 0.35s ease;
    font-size: 18px;
    pointer-events: none;
}
/* hover الأيقونة والنص */
div.stButton > button:hover::after {
    opacity: 1;
    transform: translateX(0);
}
div.stButton > button:hover > div {
    transform: translateX(-8px);
}
/* تخصيص الأيقونات */
div.stButton > button[kind="primary"]::after {
    content: "🔍";
}
div.stButton > button[kind="secondary"]::after {
    content: "🔄";
}
/* ---- bounce animation ---- */
@keyframes bounceHover {
    0% { transform: translateY(0); }
    30% { transform: translateY(-5px); }
    50% { transform: translateY(0); }
    70% { transform: translateY(-2px); }
    100% { transform: translateY(0); }
}
/* ---- primary button gradient + hover + bounce ---- */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #42a5f5, #1e88e5);
    border: none;
    color: white;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e88e5, #1565c0);
    animation: bounceHover 0.4s ease;
    box-shadow: 0 6px 15px rgba(30,136,197,0.4);
}
/* ---- secondary button gradient + hover + bounce ---- */
div.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #7b1fa2, #9c27b0);
    border: none;
    color: white;
    opacity: 1 !important;
    transition: all 0.35s ease;
}
div.stButton > button[kind="secondary"]:hover {
    background: linear-gradient(135deg, #9c27b0, #7b1fa2);
    animation: bounceHover 0.4s ease;
    box-shadow: 0 6px 15px rgba(156,39,176,0.4);
}
</style>
""", unsafe_allow_html=True)
#############################################################
#Style for loading process after uploading image
st.markdown("""
<style>
/* Overlay كامل الشاشة */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.65);
    backdrop-filter: blur(6px);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}
/* الدائرة المتحركة */
.loader {
    width: 70px;
    height: 70px;
    border: 6px solid rgba(255,255,255,0.2);
    border-top: 6px solid #4fff61;
    border-radius: 50%;
    animation: spin 2s linear infinite;
}
/* النص */
.loader-text {
    margin-top: 18px;
    font-size: 18px;
    color: #eaffea;
    letter-spacing: 0.5px;
    text-shadow: 0 0 8px rgba(79,255,97,0.8);
}
/* الحركة */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)
#############################################################
# Alert style when press check button without uploading
st.markdown("""
<style>
div[data-testid="stAlert"] {
    background: linear-gradient(135deg, #ff1744, #ff5252) !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    box-shadow: 0 0 15px rgba(255,23,68,0.8) !important;
}
div[data-testid="stAlert"] p {
    color: white !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)
#############################################################
#Header of App
st.markdown("""
<div style="text-align:center; margin-bottom: 1rem;">
  <h2 style="
      color: #f9fafb;
      font-weight: bold;
      text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
      margin-bottom: 0.3rem;
  ">
  😷 Face Mask Detection System 🤔
  </h2>
  <div style="
      width: 390px;
      height: 4px;
      background: linear-gradient(to right, #60a5fa, #2563eb);
      margin: auto;
      border-radius: 10px;
  "></div>
</div>
""", unsafe_allow_html=True)
#############################################################
#SubHeader of App
st.markdown("""
<style>
  .pro-color {
    color: #4fff61; 
    font-weight: bold;
  }
  .imp-color {
    color: #ff2b0a;
    font-weight: bold;
            
   .con-color {
    color: #b5b4d9;
  }
</style>
<p style="
    text-align: center;
    color: #b5b4d9;
    font-size: 20px;
    letter-spacing: 0.5px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.6);
    margin-top: 0.5rem;
">
🤖AI-based system for detecting usage 
<br>
✅ Mask &nbsp;&nbsp; ❌ No Mask
</p>
""", unsafe_allow_html=True)
st.markdown("""
<div style="height: 12px;"></div>
""", unsafe_allow_html=True)
#############################################################
#Image uploading part with some decorations
st.markdown("""
<style>
            
/* صندوق الرفع */
div[data-testid="stFileUploader"] {
    margin: 0 !important;
    padding: 1rem;
    border: 2px dashed #4fff61;
    border-radius: 12px;
    background: rgba(79, 255, 97, 0.05);
    transition: border-color 0.35s ease, background-color 0.35s ease;
}
            
div[data-testid="stFileUploader"]:hover {
    border-color: #22c55e;
    background-color: rgba(79, 255, 97, 0.1);
}
div[data-testid="stFileUploader"] button {
    all: unset;
    display: inline-block;
    padding: 0.5rem 1rem;
    background-color: #696969;
    color: white;
    border-radius: 8px;
    cursor: pointer;
}
            
/* اسم الملف بعد الرفع */
div[data-testid="stFileUploaderFileName"],
div[data-testid="stFileUploader"] small {
    color: #eaffea !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    text-shadow:
        0 0 6px rgba(79,255,97,0.9),
        1px 1px 2px rgba(0,0,0,0.9);
}
            
/* أيقونة الصح */
div[data-testid="stFileUploader"] svg {
    filter: drop-shadow(0 0 6px rgba(79,255,97,0.9));
}
/* Glow Animation */
@keyframes pulseGlow {
    0% { box-shadow: 0 0 6px rgba(79,255,97,.25); }
    50% { box-shadow: 0 0 14px rgba(79,255,97,.5); }
    100% { box-shadow: 0 0 6px rgba(79,255,97,.25); }
}
/* نص العنوان مع الايموجي ثابت */            
div.upload-title {
    text-align: left;
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 5px;
}
/* الايموجي ثابت اللون */
span.upload-emoji {
    color: #c4ffff;
}
/* Neon glow نص يحتوي */
span.text-gradient {
    background: none;
    -webkit-background-clip: initial;
    -webkit-text-fill-color: initial;
    
    /* Soft base color */
    color: #00ffcc; 
    font-weight: bold;
    /* We apply two animations: one for the slow pulse, one for the fast flicker */
    animation: 
        subtleNeon 3s ease-in-out infinite alternate,
        neonFlicker 3s linear infinite;
}
/* 1. The Gentle Glow Pulse */
@keyframes subtleNeon {
  from {
    text-shadow: 
      0 0 4px rgba(0, 255, 204, 0.4), 
      0 0 8px rgba(0, 255, 204, 0.2);
  }
  to {
    text-shadow: 
      0 0 6px rgba(0, 255, 204, 0.7), 
      0 0 12px rgba(0, 255, 204, 0.3);
  }
}
/* 2. The "Shop Sign" Flicker Effect */
/* This briefly cuts the opacity and shadow to simulate a loose wire */
@keyframes neonFlicker {
  0%, 19.999%, 22%, 62.999%, 64%, 64.999%, 70%, 100% {
    opacity: 1;
  }
  20%, 21.999%, 63%, 63.999%, 65%, 69.999% {
    opacity: 0.4;
    text-shadow: none; /* Light goes 'out' briefly */
  }
}
            
</style>
            
<div class="upload-title">
<span class="upload-emoji">📤</span> <span class="text-gradient">Upload Image to Check</span>
</div>
""", unsafe_allow_html=True)
#############################################################
#Uploading process
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key=0
if "loading" not in st.session_state:
    st.session_state.loading = False
loading_placeholder = st.empty()

uploaded_image = st.file_uploader("",type=["jpg","jpeg","png"],key=st.session_state.uploader_key)
image="" 
#Submit and refresh button
left, middle,right = st.columns(3)
submit=left.button("Get your answer",type="primary" ,width="stretch")
st.markdown("""
<div style="height: 12px;"></div>
""", unsafe_allow_html=True)
refresh=right.button("Try another image",type="secondary",width="stretch")

if submit and uploaded_image is not None:
    result_placeholder.markdown("""
    <div style="min-height: 10px;"></div>
    """, unsafe_allow_html=True)
    st.session_state.loading = True
    loading_placeholder.markdown("""
    <div class="loading-overlay">
        <div class="loader"></div>
        <div class="loader-text">Detecting mask... please wait</div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2)
    image=Image.open(uploaded_image).convert("RGB")
    image_arr=np.array(image)
    model = load_model()
    result=model.predict(image_arr,conf=0.5,device="cpu")
    plotted=result[0].plot()
    st.session_state.loading = False
    loading_placeholder.empty()
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&display=swap');
        /* Fade-in للصندوق */
        @keyframes fadeInCard {{
       0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
        }}
        /* Pulse للنص */
        @keyframes titlePulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.08); }}
        100% {{ transform: scale(1); }}
            }}
        /* صندوق النتائج */
    .result-card {{
    background: linear-gradient(135deg, #d3d3d3, #b2ebf2);
    border: 2.5px solid #1a1a1a;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25), 
                inset 0 0 5px rgba(0, 0, 0, 0.05);
    outline: 1px solid rgba(0, 0, 0, 0.05);
    outline-offset: -2px;
    padding: 0.8rem 1.5rem;
    border-radius: 12px;
    margin-top: 2rem;
    animation: fadeInCard 0.8s ease forwards;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
}}
/* عنوان النتائج */
.result-card h3 {{
    font-family: 'Fredoka One', sans-serif;
    font-size: 28px;
    margin: 0;
}}
/* النص فقط بالذهبي مع stroke */
.result-card h3 .result-text {{
    color: #00f2ff;
    display: inline-block;
    animation: titlePulse 1.5s ease-in-out infinite;
    will-change: transform;
    -webkit-text-stroke: 1px rgba(0, 0, 0, 0.8);
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    font-weight: bold;
    letter-spacing: 1px;
}}
    
                
.arrow-wrap {{
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.arrow-wrap img {{
    width: 26px;
    height: 26px;
}}
                
</style>
<div class="result-card">
  <div class="arrow-wrap">
  <img src="data:image/gif;base64,{arrow_gif}" />
    </div>
    <h3><span class="result-text">&nbsp;&nbsp;&nbsp;Detection Result</span></h3>
<div class="arrow-wrap">
  <img src="data:image/gif;base64,{arrow_gif}" />
 </div>
</div>
  
""", unsafe_allow_html=True)
    st.markdown("""
<style>
.safe-space {
    height: 12px;
}
</style>
<div class="safe-space"></div>
""", unsafe_allow_html=True)
    result_placeholder.empty()
    col1, col2,col3 = st.columns([1,8,1])
    col2.image(plotted,channels="RGB",width=680)
    st.markdown("""
<style>
img {
    max-width: 100%;
    height: auto;
}
</style>
""", unsafe_allow_html=True)
elif submit and uploaded_image is None:
    st.warning("⚠️ Please upload an image first")
if refresh:
    st.session_state.uploader_key+=1
    st.rerun()
#############################################################
#Footer
st.markdown(f"""
<style>
/* صندوق الفوتر */
.custom-footer {{
    background-color: rgba(192,192,192,0.3);
    padding: 1rem;
    border-radius: 17px;
    margin-top: 2rem;
    transition: box-shadow 0.35s ease, transform 0.35s ease;
}}
/* ظل خفيف عند hover على الفوتر */
.custom-footer:hover {{
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    transform: translateY(-1px);
}}
/* اسم المهندس */
.eng-color {{
    color: #facc15;
    font-weight: 600;
    pointer-events: auto;
    transition: all 0.3s ease;
}}
/* Glow عند hover فقط على اسم المهندس */
@keyframes softGlow {{
    0% {{ text-shadow: 0 0 6px rgba(250,204,21,0.6); }}
    50% {{ text-shadow: 0 0 14px rgba(250,204,21,0.9); }}
    100% {{ text-shadow: 0 0 6px rgba(250,204,21,0.6); }}
}}
.eng-color:hover {{
    animation: softGlow 0.6s ease-in-out;
}}
/* Pulse Animation للخ GIF واسم Ziad */
@keyframes subtlePulse {{
    0% {{ transform: scale(1); }}
    50% {{ transform: scale(1.08); }}
    100% {{ transform: scale(1); }}
}}
.footer-gif-hover:hover {{
    animation: subtlePulse 0.8s ease-in-out;
    transform-origin: center;
}}
/* اسم Ziad */
.ziad-name {{
    color: #75ffe6;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
/* باقي النص */
.footer-text {{
    color: #b8d8ff;
    font-size: 14px;
}}
/* Gradient متحرك على الـ hr عند hover على الفوتر */
/* Separator داخل div مع سمك أكبر */
.footer-separator {{
    height: 5px; /* سمك أكبر */
    border-radius: 4px;
    background-color: rgba(174, 163, 145, 0.7);
    margin: 1rem 0;
    transition: background 0.5s ease;
}}
/* عند hover على الفوتر، Gradient متحرك */
.custom-footer:hover .footer-separator {{
    background: linear-gradient(90deg, #75ffe6, #60a5fa, #4fd1c5, #b8d8ff);
    background-size: 200% 100%;
    animation: hrGradientMove 1s linear infinite;
}}
@keyframes hrGradientMove {{
    0% {{ background-position: 0% 0%; }}
    100% {{ background-position: 100% 0%; }}
}}
</style>
<div class="custom-footer">
<!-- Ziad -->
<p style="text-align: center; font-family: 'Dancing Script', cursive; font-size: 28px; font-weight: 600;">
    <img src="data:image/gif;base64,{footer1_gif}"
         class="footer-gif-hover"
         style="height:50px; width:50px; border-radius:50%; object-fit: cover; vertical-align:middle; margin-right:5px; border: 2px solid #d0d4db;">
    <span class="ziad-name footer-gif-hover">Ziad</span>
</p>
<div class="footer-separator"></div>
<!-- Footer Text + المهندس -->
<p class="footer-text" style="text-align: center; font-size: 14px;">
    Graduation Project – Artificial Intelligence Diploma<br>
    ©Supervised by Engineer 
    <img src="data:image/gif;base64,{footer3_gif}"
         class="footer-gif-hover"
         style="height:50px; width:50px; border-radius:50%; object-fit: cover; vertical-align:middle; margin-left:5px; margin-right:5px; border: 2px solid #d0d4db;">
    <span class="eng-color">Mohamed Bani Abdul Ghani</span>
    <img src="data:image/gif;base64,{footer2_gif}"
         class="footer-gif-hover"
         style="height:50px; width:50px; border-radius:50%; object-fit: cover; vertical-align:middle; margin-left:5px; margin-right:5px; border: 2px solid #d0d4db;">
</p>
</div>
""", unsafe_allow_html=True)