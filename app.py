import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import io
import base64

# 植生指数の計算アルゴリズム
class VegetationIndices:
    @staticmethod
    def normalize_rgb(r: float, g: float, b: float) -> tuple:
        total = r + g + b
        if total == 0:
            return (0, 0, 0)
        return (r/total, g/total, b/total)
    
    @staticmethod
    def calculate_indices(r: float, g: float, b: float) -> dict:
        nr, ng, nb = VegetationIndices.normalize_rgb(r, g, b)
        return {
            "INT": (r + g + b) / 3,
            "NRI": nr,
            "NGI": ng,
            "NBI": nb,
            "RGRI": r/g if g > 0 else 0,
            "ExR": 1.4 * r - g,
            "ExG": 2 * g - r - b,
            "ExB": 1.4 * b - g,
            "ExGR": (2 * g - r - b) - (1.4 * r - g),
            "GRVI": (g - r)/(g + r) if (g + r) > 0 else 0,
            "VARI": (g - r)/(g + r - b) if (g + r - b) != 0 else 0,
            "GLI": (2 * g - r - b)/(2 * g + r + b) if (2 * g + r + b) != 0 else 0,
            "MGRVI": (g*g - r*r)/(g*g + r*r) if (g*g + r*r) != 0 else 0,
            "RGBVI": (g*g - r*b)/(g*g + r*b) if (g*g + r*b) != 0 else 0,
            "VEG": g/(pow(r, 0.667) * pow(b, 0.333)) if (r > 0 and b > 0) else 0
        }

def calculate_otsu_threshold(image: np.ndarray) -> float:
    # ExGの計算
    b, g, r = cv2.split(image)
    total = r.astype(float) + g.astype(float) + b.astype(float)
    nr = np.divide(r, total, out=np.zeros_like(r, dtype=float), where=total!=0)
    ng = np.divide(g, total, out=np.zeros_like(g, dtype=float), where=total!=0)
    nb = np.divide(b, total, out=np.zeros_like(b, dtype=float), where=total!=0)
    
    exg = 2 * ng - nr - nb
    exg_uint8 = ((exg + 1) * 127.5).astype(np.uint8)
    
    # 大津の方法による閾値計算
    thresh, _ = cv2.threshold(exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (thresh / 127.5) - 1

def process_image(image: np.ndarray, threshold_method: str, exg_threshold: float = 0.2) -> tuple:
    # 画像の前処理
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # ExGの計算と二値化
    b, g, r = cv2.split(image)
    total = r.astype(float) + g.astype(float) + b.astype(float)
    nr = np.divide(r, total, out=np.zeros_like(r, dtype=float), where=total!=0)
    ng = np.divide(g, total, out=np.zeros_like(g, dtype=float), where=total!=0)
    nb = np.divide(b, total, out=np.zeros_like(b, dtype=float), where=total!=0)
    
    exg = 2 * ng - nr - nb
    
    # 閾値の決定
    if threshold_method == "otsu":
        threshold = calculate_otsu_threshold(image)
    else:
        threshold = exg_threshold
    
    # 二値化マスクの作成
    binary_mask = (exg >= threshold).astype(np.uint8) * 255
    
    # 植生指数の計算
    indices_veg = {"vegetation": {}, "whole": {}}
    veg_pixels = np.count_nonzero(binary_mask)
    total_pixels = binary_mask.size
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r, g, b = image[y, x] / 255.0
            indices = VegetationIndices.calculate_indices(r, g, b)
            
            # 全体の指数を更新
            for key, value in indices.items():
                indices_veg["whole"][key] = indices_veg["whole"].get(key, 0) + value
                
            # 植生部分の指数を更新
            if binary_mask[y, x] > 0:
                for key, value in indices.items():
                    indices_veg["vegetation"][key] = indices_veg["vegetation"].get(key, 0) + value
    
    # 平均値の計算
    for key in indices_veg["whole"]:
        indices_veg["whole"][key] /= total_pixels
        indices_veg["vegetation"][key] = indices_veg["vegetation"][key] / veg_pixels if veg_pixels > 0 else 0
    
    return binary_mask, veg_pixels, total_pixels, indices_veg

def main():
    st.set_page_config(page_title="Vegetation Analysis", layout="wide")
    
    st.title("Vegetation Analysis")
    
    # サイドバー設定
    st.sidebar.header("Settings")
    threshold_method = st.sidebar.radio(
        "Thresholding Method",
        ["otsu", "manual"],
        format_func=lambda x: "Otsu's Method" if x == "otsu" else "Manual Threshold"
    )
    
    exg_threshold = 0.2
    if threshold_method == "manual":
        exg_threshold = st.sidebar.slider("ExG Threshold", -1.0, 1.0, 0.2, 0.01)
    
    # ファイルアップロード
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 画像の読み込みと処理
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 画像の処理
        binary_mask, veg_pixels, total_pixels, indices = process_image(
            image, threshold_method, exg_threshold
        )
        
        # 結果の表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Vegetation Mask")
            st.image(binary_mask, use_column_width=True)
        
        # 解析結果の表示
        st.subheader("Analysis Results")
        coverage = (veg_pixels / total_pixels) * 100
        st.write(f"Vegetation Coverage: {coverage:.2f}%")
        st.write(f"Vegetation Pixels: {veg_pixels:,}")
        st.write(f"Total Pixels: {total_pixels:,}")
        
        # インデックス結果の表示
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Vegetation Indices (Vegetation Area)")
            for key, value in indices["vegetation"].items():
                st.write(f"{key}: {value:.4f}")
        
        with col4:
            st.subheader("Vegetation Indices (Whole Image)")
            for key, value in indices["whole"].items():
                st.write(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()