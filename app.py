import streamlit as st
import numpy as np
import cv2
import pandas as pd
import base64
from typing import Tuple, Dict, List
import gc

def resize_if_needed(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """大きな画像をリサイズして処理を軽くする"""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

@st.cache_data(max_entries=10)
def process_single_image(
    image: np.ndarray,
    threshold_method: str,
    exg_threshold: float,
    selected_indices: List[str]
) -> Tuple[np.ndarray, int, int, Dict]:
    """1枚の画像を処理（メモリ効率化）"""
    # 画像のリサイズ
    image = resize_if_needed(image)
    
    # float32で計算（メモリ削減）
    image_float = image.astype(np.float32) / 255.0
    b, g, r = cv2.split(image_float)
    
    # メモリ解放
    del image_float
    gc.collect()
    
    # ExG計算
    total = r + g + b
    nr = np.divide(r, total, out=np.zeros_like(r), where=total!=0)
    ng = np.divide(g, total, out=np.zeros_like(g), where=total!=0)
    nb = np.divide(b, total, out=np.zeros_like(b), where=total!=0)
    exg = 2 * ng - nr - nb
    
    # メモリ解放
    del nr, nb
    gc.collect()
    
    # 閾値処理
    if threshold_method == "otsu":
        exg_uint8 = ((exg + 1) * 127.5).astype(np.uint8)
        thresh, _ = cv2.threshold(exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = (thresh / 127.5) - 1
        del exg_uint8
    else:
        threshold = exg_threshold
    
    binary_mask = (exg >= threshold).astype(np.uint8) * 255
    del exg
    gc.collect()
    
    # 植生指数の計算（メモリ効率化）
    indices_result = {"vegetation": {}, "whole": {}}
    veg_pixels = np.count_nonzero(binary_mask)
    total_pixels = binary_mask.size
    mask_bool = binary_mask > 0
    
    for index_name in selected_indices:
        if index_name == "INT":
            value = (r + g + b) / 3
        elif index_name == "ExG":
            value = 2 * ng - r - b
        elif index_name == "GRVI":
            value = np.divide(g - r, g + r, out=np.zeros_like(g), where=(g + r)!=0)
        else:
            continue  # 必要な指数のみ計算
            
        indices_result["whole"][index_name] = float(np.mean(value))
        if veg_pixels > 0:
            indices_result["vegetation"][index_name] = float(np.mean(value[mask_bool]))
        else:
            indices_result["vegetation"][index_name] = 0.0
        
        del value
        gc.collect()
    
    return binary_mask, veg_pixels, total_pixels, indices_result

def batch_process_images(
    files: List,
    threshold_method: str,
    exg_threshold: float,
    selected_indices: List[str],
    progress_bar,
    progress_text
) -> List[Dict]:
    """バッチ処理（メモリ効率化）"""
    results = []
    total_files = len(files)
    
    for i, file in enumerate(files, 1):
        try:
            # 進捗更新
            progress = i / total_files
            progress_bar.progress(progress)
            progress_text.text(f"処理中: {file.name} ({i}/{total_files}, {progress:.1%})")
            
            # 画像読み込み
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 画像処理
            binary_mask, veg_pixels, total_pixels, indices = process_single_image(
                image, threshold_method, exg_threshold, selected_indices
            )
            
            # 結果保存
            result = {
                'filename': file.name,
                'vegetationCoverage': (veg_pixels / total_pixels) * 100,
                'vegetationPixels': veg_pixels,
                'totalPixels': total_pixels,
                'thresholdMethod': threshold_method,
                'threshold': exg_threshold if threshold_method == 'exg' else 'auto'
            }
            
            # 指数値の追加
            for key in selected_indices:
                if key in indices['vegetation']:
                    result[f'{key}_vegetation'] = indices['vegetation'][key]
                    result[f'{key}_whole'] = indices['whole'][key]
            
            results.append(result)
            
            # メモリ解放
            del image, binary_mask
            gc.collect()
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            continue
    
    return results

def main():
    st.set_page_config(page_title="Vegetation Analysis", layout="wide")
    
    # 単一画像処理
    st.header("単一画像解析")
    uploaded_file = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])
    
    # 最小限の設定オプション
    threshold_method = st.radio(
        "2値化方法",
        ["otsu", "exg"],
        format_func=lambda x: "大津の方法（自動）" if x == "otsu" else "ExGによる閾値指定"
    )
    
    exg_threshold = 0.2
    if threshold_method == "exg":
        exg_threshold = st.slider("ExG閾値", -1.0, 1.0, 0.2, 0.01)
    
    # 必要な指数のみ選択
    selected_indices = ["INT", "ExG", "GRVI"]  # 主要な指数のみ
    
    if uploaded_file:
        try:
            # 画像処理
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            binary_mask, veg_pixels, total_pixels, indices = process_single_image(
                image, threshold_method, exg_threshold, selected_indices
            )
            
            # 結果表示
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="元画像", use_column_width=True)
            with col2:
                st.image(binary_mask, caption="処理結果", use_column_width=True)
            
            # 解析結果
            coverage = (veg_pixels / total_pixels) * 100
            st.metric("植生被覆率", f"{coverage:.2f}%")
            
            # 指数結果
            for key in selected_indices:
                if key in indices['vegetation']:
                    st.write(f"{key}: {indices['vegetation'][key]:.4f}")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
    
    # バッチ処理
    st.header("バッチ処理")
    uploaded_files = st.file_uploader(
        "複数の画像をアップロード",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("バッチ処理開始"):
            progress_container = st.container()
            progress_bar = progress_container.progress(0)
            progress_text = progress_container.empty()
            
            results = batch_process_images(
                uploaded_files,
                threshold_method,
                exg_threshold,
                selected_indices,
                progress_bar,
                progress_text
            )
            
            if results:
                # 結果表示
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # CSVダウンロード
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "結果をダウンロード",
                    csv,
                    "vegetation_analysis_results.csv",
                    "text/csv"
                )

if __name__ == "__main__":
    main()