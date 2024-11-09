import streamlit as st
import numpy as np
import cv2
import pandas as pd
import base64
from typing import Tuple, Dict, List
import gc

# 植生指数の定義
ALGORITHMS = {
    "INT": ("Intensity", lambda r, g, b: (r + g + b) / 3),
    "NRI": ("Normalized Red Index", lambda r, g, b: r),
    "NGI": ("Normalized Green Index", lambda r, g, b: g),
    "NBI": ("Normalized Blue Index", lambda r, g, b: b),
    "RGRI": ("Red Green Ratio Index", lambda r, g, b: np.divide(r, g, out=np.zeros_like(r), where=g!=0)),
    "ExR": ("Excess Red Index", lambda r, g, b: 1.4 * r - g),
    "ExG": ("Excess Green Index", lambda r, g, b: 2 * g - r - b),
    "ExB": ("Excess Blue Index", lambda r, g, b: 1.4 * b - g),
    "ExGR": ("Excess Green minus Red Index", lambda r, g, b: (2 * g - r - b) - (1.4 * r - g)),
    "GRVI": ("Green Red Vegetation Index", lambda r, g, b: np.divide(g - r, g + r, out=np.zeros_like(r), where=(g + r)!=0)),
    "VARI": ("Visible Atmospherically Resistant Index", lambda r, g, b: np.divide(g - r, g + r - b, out=np.zeros_like(r), where=(g + r - b)!=0)),
    "GLI": ("Green Leaf Index", lambda r, g, b: np.divide(2 * g - r - b, 2 * g + r + b, out=np.zeros_like(r), where=(2 * g + r + b)!=0)),
    "MGRVI": ("Modified Green Red Vegetation Index", lambda r, g, b: np.divide(g*g - r*r, g*g + r*r, out=np.zeros_like(r), where=(g*g + r*r)!=0)),
    "RGBVI": ("Red Green Blue Vegetation Index", lambda r, g, b: np.divide(g*g - r*b, g*g + r*b, out=np.zeros_like(r), where=(g*g + r*b)!=0)),
    "VEG": ("Vegetativen", lambda r, g, b: np.divide(g, np.power(r, 0.667) * np.power(b, 0.333), out=np.zeros_like(r), where=(r > 0) & (b > 0)))
}

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
    
    # 正規化
    total = r + g + b
    nr = np.divide(r, total, out=np.zeros_like(r), where=total!=0)
    ng = np.divide(g, total, out=np.zeros_like(g), where=total!=0)
    nb = np.divide(b, total, out=np.zeros_like(b), where=total!=0)
    
    # ExG計算と二値化
    exg = 2 * ng - nr - nb
    
    if threshold_method == "otsu":
        exg_uint8 = ((exg + 1) * 127.5).astype(np.uint8)
        thresh, _ = cv2.threshold(exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = (thresh / 127.5) - 1
        del exg_uint8
    else:
        threshold = exg_threshold
    
    binary_mask = (exg >= threshold).astype(np.uint8) * 255
    
    # 植生指数の計算
    indices_result = {"vegetation": {}, "whole": {}}
    veg_pixels = np.count_nonzero(binary_mask)
    total_pixels = binary_mask.size
    mask_bool = binary_mask > 0
    
    # 選択された指数の計算（ベクトル化処理）
    for index_name in selected_indices:
        if index_name in ALGORITHMS:
            value = ALGORITHMS[index_name][1](nr, ng, nb)
            indices_result["whole"][index_name] = float(np.mean(value))
            if veg_pixels > 0:
                indices_result["vegetation"][index_name] = float(np.mean(value[mask_bool]))
            else:
                indices_result["vegetation"][index_name] = 0.0
            del value
    
    return binary_mask, veg_pixels, total_pixels, indices_result

def main():
    st.set_page_config(page_title="Vegetation Analysis", layout="wide")
    
    st.title("植生解析アプリケーション")
    
    # サイドバーでの設定
    with st.sidebar:
        st.header("解析設定")
        
        # 2値化方法の選択
        threshold_method = st.radio(
            "2値化方法",
            ["otsu", "exg"],
            format_func=lambda x: "大津の方法（自動）" if x == "otsu" else "ExGによる閾値指定"
        )
        
        if threshold_method == "exg":
            exg_threshold = st.slider("ExG閾値", -1.0, 1.0, 0.2, 0.01)
        else:
            exg_threshold = 0.2
        
        # 植生指数の選択
        st.subheader("使用する植生指数")
        selected_indices = []
        indices_columns = st.columns(2)
        for i, (key, (name, _)) in enumerate(ALGORITHMS.items()):
            with indices_columns[i % 2]:
                if st.checkbox(f"{key} - {name}", value=key in ["ExG", "GRVI"]):
                    selected_indices.append(key)
    
    # メインコンテンツ
    tab1, tab2 = st.tabs(["単一画像解析", "バッチ処理"])
    
    with tab1:
        uploaded_file = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            try:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                binary_mask, veg_pixels, total_pixels, indices = process_single_image(
                    image, threshold_method, exg_threshold, selected_indices
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="元画像")
                with col2:
                    st.image(binary_mask, caption="植生抽出結果")
                
                # 結果表示
                coverage = (veg_pixels / total_pixels) * 100
                
                metrics_cols = st.columns(3)
                with metrics_cols[0]:
                    st.metric("植生被覆率", f"{coverage:.2f}%")
                with metrics_cols[1]:
                    st.metric("植生ピクセル数", f"{veg_pixels:,}")
                with metrics_cols[2]:
                    st.metric("総ピクセル数", f"{total_pixels:,}")
                
                if indices["vegetation"]:
                    st.subheader("植生指数の計算結果")
                    index_cols = st.columns(2)
                    with index_cols[0]:
                        st.write("植生部分の指数値:")
                        for key in selected_indices:
                            st.write(f"{ALGORITHMS[key][0]}: {indices['vegetation'][key]:.4f}")
                    with index_cols[1]:
                        st.write("画像全体の指数値:")
                        for key in selected_indices:
                            st.write(f"{ALGORITHMS[key][0]}: {indices['whole'][key]:.4f}")
            
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
    
    with tab2:
        uploaded_files = st.file_uploader(
            "複数の画像をアップロード",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )
        
        if uploaded_files and len(selected_indices) > 0:
            if st.button("バッチ処理開始", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                total_files = len(uploaded_files)
                
                for i, file in enumerate(uploaded_files, 1):
                    try:
                        progress = i / total_files
                        progress_bar.progress(progress)
                        status_text.text(f"処理中: {file.name} ({i}/{total_files}, {progress:.1%})")
                        
                        file_bytes = np.frombuffer(file.read(), np.uint8)
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        binary_mask, veg_pixels, total_pixels, indices = process_single_image(
                            image, threshold_method, exg_threshold, selected_indices
                        )
                        
                        result = {
                            'ファイル名': file.name,
                            '植生被覆率(%)': (veg_pixels / total_pixels) * 100,
                            '植生ピクセル数': veg_pixels,
                            '総ピクセル数': total_pixels,
                            '2値化方法': "大津の方法" if threshold_method == "otsu" else "ExG閾値指定",
                            '閾値': "自動" if threshold_method == "otsu" else str(exg_threshold)
                        }
                        
                        for key in selected_indices:
                            result[f'{ALGORITHMS[key][0]}(植生部)'] = indices['vegetation'][key]
                            result[f'{ALGORITHMS[key][0]}(全体)'] = indices['whole'][key]
                        
                        results.append(result)
                        
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                        continue
                
                if results:
                    status_text.text("処理完了!")
                    
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                    
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "CSVファイルをダウンロード",
                        csv,
                        "vegetation_analysis_results.csv",
                        "text/csv",
                        key='download-csv'
                    )

if __name__ == "__main__":
    main()