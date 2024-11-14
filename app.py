import streamlit as st
import numpy as np
import cv2
import pandas as pd
import base64
from typing import Tuple, Dict, List
import gc

# 言語定義
TRANSLATIONS = {
    "en": {
        "app_title": "Vegetation Analysis Application",
        "analysis_settings": "Analysis Settings",
        "threshold_method": "Thresholding Method",
        "otsu_method": "Otsu's Method (Automatic)",
        "exg_threshold_method": "ExG Threshold",
        "exg_threshold_value": "ExG Threshold Value",
        "vegetation_indices": "Vegetation Indices",
        "single_image": "Single Image Analysis",
        "batch_processing": "Batch Processing",
        "upload_image": "Upload Image",
        "upload_multiple": "Upload Multiple Images",
        "original_image": "Original Image",
        "vegetation_result": "Vegetation Extraction Result",
        "coverage_rate": "Vegetation Coverage",
        "veg_pixels": "Vegetation Pixels",
        "total_pixels": "Total Pixels",
        "index_results": "Vegetation Index Results",
        "veg_area_values": "Values for Vegetation Area:",
        "whole_area_values": "Values for Whole Image:",
        "error_occurred": "An error occurred: ",
        "start_batch": "Start Batch Processing",
        "processing": "Processing: ",
        "processing_complete": "Processing Complete!",
        "download_csv": "Download CSV File",
        "file_name": "File Name",
        "coverage_rate_percent": "Vegetation Coverage (%)",
        "threshold_method_col": "Threshold Method",
        "threshold_value": "Threshold Value",
        "automatic": "Automatic",
            "help": "Help",
        "about_title": "About This Application",
        "about_description": """
        This application analyzes vegetation in images using various vegetation indices. You can process both single images and multiple images in batch mode.
        
        **Key Features:**
        - Vegetation extraction using ExG
        - Calculation of multiple vegetation indices
        - Support for batch processing of multiple images
        - Detailed analysis results with CSV export
        
        **How to Use:**
        1. **Analysis Settings (Sidebar)**
           - Select thresholding method (Otsu's or ExG)
           - If using ExG, adjust the threshold value
           - Choose vegetation indices to calculate
        
        2. **Single Image Analysis**
           - Upload a single image
           - View the original and processed images
           - See vegetation coverage and index calculations
        
        3. **Batch Processing**
           - Upload multiple images
           - Start processing to analyze all images
           - Download results as CSV
        
        **Tips:**
        - For best results, use clear images with good contrast
        - The ExG threshold can be adjusted if the default value doesn't provide good results
        - Batch processing is ideal for analyzing multiple images with the same settings
        """,    
    },
    "es": {
        "app_title": "Aplicación de Análisis de Vegetación",
        "analysis_settings": "Configuración de Análisis",
        "threshold_method": "Método de Umbral",
        "otsu_method": "Método de Otsu (Automático)",
        "exg_threshold_method": "Umbral ExG",
        "exg_threshold_value": "Valor del Umbral ExG",
        "vegetation_indices": "Índices de Vegetación",
        "single_image": "Análisis de Imagen Individual",
        "batch_processing": "Procesamiento por Lotes",
        "upload_image": "Subir Imagen",
        "upload_multiple": "Subir Múltiples Imágenes",
        "original_image": "Imagen Original",
        "vegetation_result": "Resultado de Extracción de Vegetación",
        "coverage_rate": "Cobertura de Vegetación",
        "veg_pixels": "Píxeles de Vegetación",
        "total_pixels": "Píxeles Totales",
        "index_results": "Resultados de Índices de Vegetación",
        "veg_area_values": "Valores para Área de Vegetación:",
        "whole_area_values": "Valores para Imagen Completa:",
        "error_occurred": "Ocurrió un error: ",
        "start_batch": "Iniciar Procesamiento por Lotes",
        "processing": "Procesando: ",
        "processing_complete": "¡Procesamiento Completado!",
        "download_csv": "Descargar Archivo CSV",
        "file_name": "Nombre del Archivo",
        "coverage_rate_percent": "Cobertura de Vegetación (%)",
        "threshold_method_col": "Método de Umbral",
        "threshold_value": "Valor del Umbral",
        "automatic": "Automático",
            "help": "Ayuda",
        "about_title": "Acerca de esta Aplicación",
        "about_description": """
        Esta aplicación analiza la vegetación en imágenes utilizando varios índices de vegetación. Puede procesar tanto imágenes individuales como múltiples imágenes en modo por lotes.
        
        **Características Principales:**
        - Extracción de vegetación usando ExG
        - Cálculo de múltiples índices de vegetación
        - Soporte para procesamiento por lotes de múltiples imágenes
        - Resultados detallados con exportación a CSV
        
        **Cómo Usar:**
        1. **Configuración de Análisis (Barra Lateral)**
           - Seleccione el método de umbral (Otsu o ExG)
           - Si usa ExG, ajuste el valor del umbral
           - Elija los índices de vegetación a calcular
        
        2. **Análisis de Imagen Individual**
           - Suba una imagen
           - Vea la imagen original y procesada
           - Observe la cobertura vegetal y los cálculos de índices
        
        3. **Procesamiento por Lotes**
           - Suba múltiples imágenes
           - Inicie el procesamiento para analizar todas las imágenes
           - Descargue los resultados en CSV
        
        **Consejos:**
        - Para mejores resultados, use imágenes claras con buen contraste
        - El umbral ExG puede ajustarse si el valor predeterminado no proporciona buenos resultados
        - El procesamiento por lotes es ideal para analizar múltiples imágenes con la misma configuración
        """,    
    },
    "ja": {
        "app_title": "植生解析アプリケーション",
        "analysis_settings": "解析設定",
        "threshold_method": "2値化方法",
        "otsu_method": "大津の方法（自動）",
        "exg_threshold_method": "ExGによる閾値指定",
        "exg_threshold_value": "ExG閾値",
        "vegetation_indices": "使用する植生指数",
        "single_image": "単一画像解析",
        "batch_processing": "バッチ処理",
        "upload_image": "画像をアップロード",
        "upload_multiple": "複数の画像をアップロード",
        "original_image": "元画像",
        "vegetation_result": "植生抽出結果",
        "coverage_rate": "植生被覆率",
        "veg_pixels": "植生ピクセル数",
        "total_pixels": "総ピクセル数",
        "index_results": "植生指数の計算結果",
        "veg_area_values": "植生部分の指数値:",
        "whole_area_values": "画像全体の指数値:",
        "error_occurred": "エラーが発生しました: ",
        "start_batch": "バッチ処理開始",
        "processing": "処理中: ",
        "processing_complete": "処理完了!",
        "download_csv": "CSVファイルをダウンロード",
        "file_name": "ファイル名",
        "coverage_rate_percent": "植生被覆率(%)",
        "threshold_method_col": "2値化方法",
        "threshold_value": "閾値",
        "automatic": "自動",
            "help": "ヘルプ",
        "about_title": "このアプリケーションについて",
        "about_description": """
        このアプリケーションは、様々な植生指数を用いて画像内の植生を解析します。単一画像の処理と複数画像の一括処理の両方に対応しています。

        **主な機能：**
        - ExGによる植生のある領域の抽出
        - 複数の植生指数の計算
        - 複数画像の一括処理機能
        - 詳細な解析結果のCSVエクスポート

        **使用方法：**
        1. **解析設定（サイドバー）**
           - 2値化方法の選択（大津の方法またはExG）
           - ExGを使用する場合は閾値を調整
           - 計算する植生指数を選択

        2. **単一画像解析**
           - 画像をアップロード
           - 元画像と処理結果を確認
           - 植生被覆率と指数計算結果を確認

        3. **バッチ処理**
           - 複数の画像をアップロード
           - 処理を開始して全画像を解析
           - 結果をCSVでダウンロード

        **ヒント：**
        - より良い結果を得るために、コントラストの良い鮮明な画像を使用してください
        - デフォルトの値で良い結果が得られない場合はExGの閾値を調整してください
        - 同じ設定で複数の画像を解析する場合はバッチ処理が便利です
        """,    
    }
}

# 植生指数の定義
ALGORITHMS = {
    "INT": ("Intensity", lambda r, g, b: (r + g + b)/255 / 3),
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

def get_text(key: str, lang: str) -> str:
    """指定された言語のテキストを取得"""
    return TRANSLATIONS[lang][key]

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, Dict, float]:
    """1枚の画像を処理（メモリ効率化）"""
    # 画像のリサイズ
    image = resize_if_needed(image)
    
    # float32で計算（メモリ削減）
    image_float = image.astype(np.float32) 
    b, g, r = cv2.split(image_float)
    
    # メモリ解放
    del image_float
    gc.collect()
    
    # 正規化 (INTは除外)
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
    
    # マスクされた画像の作成
    masked_image = image.copy()
    masked_image[binary_mask == 0] = 0
    
    # エッジ検出
    edges = cv2.Canny(binary_mask, 100, 200)
    
    # Perimeter Area Ratio (PAR)の計算
    perimeter_pixels = np.count_nonzero(edges)
    veg_pixels = np.count_nonzero(binary_mask)
    par = perimeter_pixels / veg_pixels if veg_pixels > 0 else 0
    
    # 植生指数の計算
    indices_result = {"vegetation": {}, "whole": {}}
    total_pixels = binary_mask.size
    mask_bool = binary_mask > 0
    
    # 選択された指数の計算（ベクトル化処理）
    for index_name in selected_indices:
        if index_name == "INT":
            # INTは正規化前の値を使用
            value = ALGORITHMS[index_name][1](r, g, b)
        elif index_name in ALGORITHMS:
            value = ALGORITHMS[index_name][1](nr, ng, nb)
            
        indices_result["whole"][index_name] = float(np.mean(value))
        if veg_pixels > 0:
            indices_result["vegetation"][index_name] = float(np.mean(value[mask_bool]))
        else:
            indices_result["vegetation"][index_name] = 0.0
        del value
    
    return binary_mask, masked_image, edges, veg_pixels, total_pixels, indices_result, par


def main():
    st.set_page_config(page_title="Vegetation Analysis", layout="wide")
    
    # 言語選択
    lang = st.sidebar.selectbox(
        "Language / Idioma / 言語",
        ["en", "es", "ja"],
        format_func=lambda x: {"en": "English", "es": "Español", "ja": "日本語"}[x]
    )
    
    st.title(get_text("app_title", lang))
    
    # ヘルプボタンを追加
    if st.button(get_text("help", lang), type="secondary"):
        st.markdown("## " + get_text("about_title", lang))
        st.markdown(get_text("about_description", lang))
        st.divider()
    
    # サイドバーでの設定
    with st.sidebar:
        st.header(get_text("analysis_settings", lang))
        
        threshold_method = st.radio(
            get_text("threshold_method", lang),
            ["otsu", "exg"],
            format_func=lambda x: get_text("otsu_method" if x == "otsu" else "exg_threshold_method", lang)
        )
        
        if threshold_method == "exg":
            exg_threshold = st.slider(get_text("exg_threshold_value", lang), -1.0, 1.0, 0.2, 0.01)
        else:
            exg_threshold = 0.2
        
        st.subheader(get_text("vegetation_indices", lang))
        selected_indices = []
        indices_columns = st.columns(2)
        for i, (key, (name, _)) in enumerate(ALGORITHMS.items()):
            with indices_columns[i % 2]:
                if st.checkbox(f"{key} - {name}", value=key in ["ExG", "GRVI","VARI"]):
                    selected_indices.append(key)
    
    tab1, tab2 = st.tabs([
        get_text("single_image", lang),
        get_text("batch_processing", lang)
    ])
    
    with tab1:
        uploaded_file = st.file_uploader(
            get_text("upload_image", lang),
            type=["png", "jpg", "jpeg"]
        )
        
    with tab1:
        uploaded_file = st.file_uploader(
            get_text("upload_image", lang),
            type=["png", "jpg", "jpeg"]
        )
        
        if uploaded_file:
            try:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                binary_mask, masked_image, edges, veg_pixels, total_pixels, indices, par = process_single_image(
                    image, threshold_method, exg_threshold, selected_indices
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(image, caption=get_text("original_image", lang))
                with col2:
                    st.image(masked_image, caption=get_text("vegetation_result", lang))
                with col3:
                    st.image(edges, caption="Edge Detection Result")
                
                coverage = (veg_pixels / total_pixels) * 100
                
                metrics_cols = st.columns(4)  # 4列に変更
                with metrics_cols[0]:
                    st.metric(get_text("coverage_rate", lang), f"{coverage:.2f}%")
                with metrics_cols[1]:
                    st.metric(get_text("veg_pixels", lang), f"{veg_pixels:,}")
                with metrics_cols[2]:
                    st.metric(get_text("total_pixels", lang), f"{total_pixels:,}")
                with metrics_cols[3]:
                    st.metric("PAR (Perimeter Area Ratio)", f"{par:.4f}")
                
                if indices["vegetation"]:
                    st.subheader(get_text("index_results", lang))
                    index_cols = st.columns(2)
                    with index_cols[0]:
                        st.write(get_text("veg_area_values", lang))
                        for key in selected_indices:
                            st.write(f"{ALGORITHMS[key][0]}: {indices['vegetation'][key]:.4f}")
                    with index_cols[1]:
                        st.write(get_text("whole_area_values", lang))
                        for key in selected_indices:
                            st.write(f"{ALGORITHMS[key][0]}: {indices['whole'][key]:.4f}")
            
            except Exception as e:
                st.error(f"{get_text('error_occurred', lang)} {str(e)}")
    
    with tab2:
        uploaded_files = st.file_uploader(
            get_text("upload_multiple", lang),
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )
        
        if uploaded_files and len(selected_indices) > 0:
            if st.button(get_text("start_batch", lang), type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                total_files = len(uploaded_files)
                
                for i, file in enumerate(uploaded_files, 1):
                    try:
                        progress = i / total_files
                        progress_bar.progress(progress)
                        status_text.text(f"{get_text('processing', lang)} {file.name} ({i}/{total_files}, {progress:.1%})")
                        
                        # バッチ処理部分は変更なし
                        file_bytes = np.frombuffer(file.read(), np.uint8)
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        binary_mask, veg_pixels, total_pixels, indices = process_single_image(
                            image, threshold_method, exg_threshold, selected_indices
                        )
                        
                        result = {
                            get_text("file_name", lang): file.name,
                            get_text("coverage_rate_percent", lang): (veg_pixels / total_pixels) * 100,
                            get_text("veg_pixels", lang): veg_pixels,
                            get_text("total_pixels", lang): total_pixels,
                            get_text("threshold_method_col", lang): get_text("otsu_method" if threshold_method == "otsu" else "exg_threshold_method", lang),
                            get_text("threshold_value", lang): get_text("automatic", lang) if threshold_method == "otsu" else str(exg_threshold)
                        }
                        
                        for key in selected_indices:
                            result[f'{ALGORITHMS[key][0]}({get_text("veg_area_values", lang)})'] = indices['vegetation'][key]
                            result[f'{ALGORITHMS[key][0]}({get_text("whole_area_values", lang)})'] = indices['whole'][key]
                        
                        results.append(result)
                        
                    except Exception as e:
                        st.error(f"{get_text('error_occurred', lang)} {str(e)}")
                        continue
                
                if results:
                    status_text.text(get_text("processing_complete", lang))
                    
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                    
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        get_text("download_csv", lang),
                        csv,
                        "vegetation_analysis_results.csv",
                        "text/csv",
                        key='download-csv'
                    )

if __name__ == "__main__":
    main()