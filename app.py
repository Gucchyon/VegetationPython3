import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import io
import base64
from functools import lru_cache
from datetime import datetime  # ファイル名用のタイムスタンプ

# キャッシュを使用して頻繁に呼び出される関数を最適化
@st.cache_data
def init_session_state():
    if 'language' not in st.session_state:
        st.session_state.language = 'ja'
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
# 植生指数の計算アルゴリズム定義
ALGORITHMS = {
    "INT": "Intensity",
    "NRI": "Normalized Red Index",
    "NGI": "Normalized Green Index",
    "NBI": "Normalized Blue Index",
    "RGRI": "Red Green Ratio Index",
    "ExR": "Excess Red Index",
    "ExG": "Excess Green Index",
    "ExB": "Excess Blue Index",
    "ExGR": "Excess Green minus Red Index",
    "GRVI": "Green Red Vegetation Index",
    "VARI": "Visible Atmospherically Resistant Index",
    "GLI": "Green Leaf Index",
    "MGRVI": "Modified Green Red Vegetation Index",
    "RGBVI": "Red Green Blue Vegetation Index",
    "VEG": "Vegetativen"
}
# 翻訳辞書は変更なし
TRANSLATIONS = {
    'ja': {
        'title': "植生の解析",
        'thresholdMethod': {
            'label': "2値化方法",
            'otsu': "大津の方法（自動）",
            'exg': "ExGによる閾値指定"
        },
        'threshold': "ExG閾値",
        'algorithm': {
            'label': "植生指数"
        },
        'singleAnalysis': "単一画像解析",
        'batchProcessing': {
            'title': "バッチ処理",
            'start': "バッチ処理開始",
            'processing': "処理中..."
        },
        'images': {
            'original': "元画像",
            'processed': "2値化画像"
        },
        'results': {
            'title': "解析結果",
            'coverage': "植生被覆率",
            'vegetationPixels': "植生ピクセル数",
            'totalPixels': "総ピクセル数",
            'indices': "植生指数値",
            'vegetationIndices': "植生部分の指数値",
            'wholeIndices': "画像全体の指数値"
        },
        'description': {
            'title': "このツールについて",
            'overview': {
                'title': "概要",
                'content': "このツールは、画像から植生領域を抽出し、様々な植生指数を計算します。"
            },
            'usage': {
                'title': "使い方",
                'steps': [
                    "使用したい植生指数のチェックボックスを選択します",
                    "2値化方法を選択します（大津の方法：自動、ExG：手動閾値設定）",
                    "画像をアップロードすると自動で解析が開始されます",
                    "複数の画像を一括処理する場合は、バッチ処理機能を使用してください"
                ]
            }
        },
        'download': "CSVダウンロード",
        'language': "English"
    },
    'en': {
        'title': "Vegetation Analysis",
        'thresholdMethod': {
            'label': "Thresholding Method",
            'otsu': "Otsu's Method (Automatic)",
            'exg': "ExG Threshold"
        },
        'threshold': "ExG Threshold",
        'algorithm': {
            'label': "Vegetation Indices"
        },
        'singleAnalysis': "Single Image Analysis",
        'batchProcessing': {
            'title': "Batch Processing",
            'start': "Start Batch Processing",
            'processing': "Processing..."
        },
        'images': {
            'original': "Original Image",
            'processed': "Binary Image"
        },
        'results': {
            'title': "Analysis Results",
            'coverage': "Vegetation Coverage",
            'vegetationPixels': "Vegetation Pixels",
            'totalPixels': "Total Pixels",
            'indices': "Vegetation Indices",
            'vegetationIndices': "Indices (Vegetation Area)",
            'wholeIndices': "Indices (Whole Image)"
        },
        'description': {
            'title': "About This Tool",
            'overview': {
                'title': "Overview",
                'content': "This tool extracts vegetation areas from images and calculates various vegetation indices."
            },
            'usage': {
                'title': "How to Use",
                'steps': [
                    "Select the vegetation indices you want to calculate",
                    "Choose the thresholding method (Otsu: automatic, ExG: manual threshold)",
                    "Upload an image to start automatic analysis",
                    "For multiple images, use the batch processing feature"
                ]
            }
        },
        'download': "Download CSV",
        'language': "日本語"
    }
}

# 植生指数の計算をベクトル化
class VegetationIndices:
    @staticmethod
    @np.vectorize
    def safe_divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b!=0)
    
    @staticmethod
    def calculate_indices_vectorized(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> dict:
        total = r + g + b
        nr = VegetationIndices.safe_divide(r, total)
        ng = VegetationIndices.safe_divide(g, total)
        nb = VegetationIndices.safe_divide(b, total)
        
        indices = {}
        indices["INT"] = (r + g + b) / 3
        indices["NRI"] = nr
        indices["NGI"] = ng
        indices["NBI"] = nb
        indices["RGRI"] = VegetationIndices.safe_divide(r, g)
        indices["ExR"] = 1.4 * r - g
        indices["ExG"] = 2 * g - r - b
        indices["ExB"] = 1.4 * b - g
        indices["ExGR"] = (2 * g - r - b) - (1.4 * r - g)
        indices["GRVI"] = VegetationIndices.safe_divide(g - r, g + r)
        indices["VARI"] = VegetationIndices.safe_divide(g - r, g + r - b)
        indices["GLI"] = VegetationIndices.safe_divide(2 * g - r - b, 2 * g + r + b)
        
        g2 = g * g
        r2 = r * r
        rb = r * b
        indices["MGRVI"] = VegetationIndices.safe_divide(g2 - r2, g2 + r2)
        indices["RGBVI"] = VegetationIndices.safe_divide(g2 - rb, g2 + rb)
        
        # VEGは特殊な計算が必要なため個別に処理
        mask = (r > 0) & (b > 0)
        veg = np.zeros_like(r)
        veg[mask] = g[mask] / (np.power(r[mask], 0.667) * np.power(b[mask], 0.333))
        indices["VEG"] = veg
        
        return indices

@st.cache_data
def calculate_otsu_threshold(image: np.ndarray) -> float:
    """大津の方法による閾値計算（高速化）"""
    b, g, r = cv2.split(image)
    total = r.astype(float) + g.astype(float) + b.astype(float)
    nr = np.divide(r, total, out=np.zeros_like(r, dtype=float), where=total!=0)
    ng = np.divide(g, total, out=np.zeros_like(g, dtype=float), where=total!=0)
    nb = np.divide(b, total, out=np.zeros_like(b, dtype=float), where=total!=0)
    
    exg = 2 * ng - nr - nb
    exg_uint8 = ((exg + 1) * 127.5).astype(np.uint8)
    thresh, _ = cv2.threshold(exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return (thresh / 127.5) - 1

@st.cache_data
def process_image(image: np.ndarray, threshold_method: str, exg_threshold: float, selected_indices: list) -> tuple:
    """画像処理のメイン関数（最適化版）"""
    # グレースケール画像の処理
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 画像の正規化（一度に実行）
    image_float = image.astype(np.float32) / 255.0
    b, g, r = cv2.split(image_float)
    
    # ExGの計算（ベクトル化）
    total = r + g + b
    nr = np.divide(r, total, out=np.zeros_like(r), where=total!=0)
    ng = np.divide(g, total, out=np.zeros_like(g), where=total!=0)
    nb = np.divide(b, total, out=np.zeros_like(b), where=total!=0)
    exg = 2 * ng - nr - nb
    
    # 閾値処理
    threshold = calculate_otsu_threshold(image) if threshold_method == "otsu" else exg_threshold
    binary_mask = (exg >= threshold).astype(np.uint8) * 255
    
    # 植生指数の計算（ベクトル化）
    indices = VegetationIndices.calculate_indices_vectorized(r, g, b)
    
    # 結果の集計
    veg_pixels = np.count_nonzero(binary_mask)
    total_pixels = binary_mask.size
    
    # 選択された指数の処理
    indices_result = {"vegetation": {}, "whole": {}}
    mask_bool = binary_mask > 0
    
    for index_name in selected_indices:
        index_values = indices[index_name]
        indices_result["whole"][index_name] = np.mean(index_values)
        indices_result["vegetation"][index_name] = np.mean(index_values[mask_bool]) if veg_pixels > 0 else 0
    
    return binary_mask, veg_pixels, total_pixels, indices_result

@st.cache_data
def create_csv_download_link(df: pd.DataFrame) -> str:
    """CSVダウンロードリンクの作成（キャッシュ対応）"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'data:text/csv;base64,{b64}'

def main():
    st.set_page_config(page_title="Vegetation Analysis", layout="wide")
    init_session_state()
    
    # 言語設定
    t = TRANSLATIONS[st.session_state.language]
    
    # ヘッダー部分
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title(t['title'])
    with col2:
        if st.button(t['language']):
            st.session_state.language = 'en' if st.session_state.language == 'ja' else 'ja'
            st.rerun()
    
    # 説明部分
    with st.expander(t['description']['title']):
        st.subheader(t['description']['overview']['title'])
        st.write(t['description']['overview']['content'])
        st.subheader(t['description']['usage']['title'])
        for step in t['description']['usage']['steps']:
            st.write(f"- {step}")
    
    # サイドバー設定
    with st.sidebar:
        # 植生指数の選択
        st.subheader(t['algorithm']['label'])
        selected_indices = {}
        for key in ALGORITHMS.keys():
            selected_indices[key] = st.checkbox(key, value=True)
        
        # 2値化方法の選択
        st.subheader(t['thresholdMethod']['label'])
        threshold_method = st.radio(
            "",
            ["otsu", "exg"],
            format_func=lambda x: t['thresholdMethod']['otsu'] if x == "otsu" else t['thresholdMethod']['exg']
        )
        
        if threshold_method == "exg":
            exg_threshold = st.slider(t['threshold'], -1.0, 1.0, 0.2, 0.01)
        else:
            exg_threshold = 0.2
    
    # メインコンテンツ
    st.header(t['singleAnalysis'])
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # 画像の読み込みと処理
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 画像の処理
        binary_mask, veg_pixels, total_pixels, indices = process_image(
            image, threshold_method, exg_threshold,
            [k for k, v in selected_indices.items() if v]
        )
        
        # 結果の表示
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(t['images']['original'])
            st.image(image)
        with col2:
            st.subheader(t['images']['processed'])
            st.image(binary_mask)
        
        # 解析結果の表示
        st.header(t['results']['title'])
        coverage = (veg_pixels / total_pixels) * 100
        
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric(t['results']['coverage'], f"{coverage:.2f}%")
        with col4:
            st.metric(t['results']['vegetationPixels'], f"{veg_pixels:,}")
        with col5:
            st.metric(t['results']['totalPixels'], f"{total_pixels:,}")
        
        # 指数結果の表示
        col6, col7 = st.columns(2)
        with col6:
            st.subheader(t['results']['vegetationIndices'])
            for key, value in indices["vegetation"].items():
                if selected_indices[key]:
                    st.write(f"{key}: {value:.4f}")
        
        with col7:
            st.subheader(t['results']['wholeIndices'])
            for key, value in indices["whole"].items():
                if selected_indices[key]:
                    st.write(f"{key}: {value:.4f}")
    
    # バッチ処理セクション
    st.header(t['batchProcessing']['title'])
    uploaded_files = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if st.button(t['batchProcessing']['start']):
            # プログレスバーと進捗テキストを表示するコンテナを作成
            progress_container = st.container()
            progress_bar = progress_container.progress(0)
            progress_text = progress_container.empty()
            
            results = []
            total_files = len(uploaded_files)
            
            for i, file in enumerate(uploaded_files):
                try:
                    # 進捗率の計算と表示
                    progress = (i + 1) / total_files
                    progress_bar.progress(progress)
                    progress_text.text(f"処理中: {file.name} ({i + 1}/{total_files}, {progress * 100:.1f}%)")
                    
                    # 画像の読み込みと処理
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # 画像の処理
                    binary_mask, veg_pixels, total_pixels, indices = process_image(
                        image, threshold_method, exg_threshold,
                        [k for k, v in selected_indices.items() if v]
                    )
                    
                    # 結果の保存
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
                        if selected_indices[key]:
                            result[f'{key}_vegetation'] = indices['vegetation'][key]
                            result[f'{key}_whole'] = indices['whole'][key]
                    
                    results.append(result)
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            # 処理完了の表示
            progress_text.text("処理完了!")
            
            # 結果の表示
            if results:
                st.success(f"処理完了: {len(results)}/{total_files} ファイルを処理しました")
                
                # 結果をDataFrameに変換
                df = pd.DataFrame(results)
                
                # CSVダウンロードボタンの作成
                csv = df.to_csv(index=False).encode('utf-8-sig')  # UTF-8 with BOM for Excel
                st.download_button(
                    label=t["download"],
                    data=csv,
                    file_name=f"vegetation_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # 結果の表示
                st.dataframe(df)
            else:
                st.warning("処理可能なファイルがありませんでした")

if __name__ == "__main__":
    main()