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
        - Vegetation extraction using Excess Green Index (ExG) with either automatic (Otsu) or manual thresholding
        - Multiple calculation modes:
          - Raw values: Calculated from the original image
          - Masked values: Reduces the influence of background soil and water by focusing only on vegetation areas
        - Edge detection and Perimeter Area Ratio (PAR) calculation to evaluate leaf thinness and size
        - Support for batch processing of multiple images
        - Detailed analysis results with CSV export
        
        **How to Use:**
        1. **Analysis Settings (Sidebar)**
           - Select ExG thresholding method: 
             - Otsu's method for automatic threshold determination
             - Manual threshold setting for fine-tuning
           - Choose vegetation indices to calculate
        
        2. **Single Image Analysis**
           - Upload a single image
           - View the original image, vegetation mask, and edge detection results
           - See vegetation coverage and PAR (higher PAR indicates thinner or smaller leaves)
           - Compare indices between raw image and masked areas
        
        3. **Batch Processing**
           - Upload multiple images
           - Start processing to analyze all images
           - Download comprehensive results as CSV
        
        **Tips:**
        - For best results, use clear images with good contrast
        - Use Otsu's method first, then switch to manual threshold if needed
        - Higher PAR values indicate thinner leaf structure or smaller leaves
        - Masked values are useful when background (soil, water) might affect the analysis
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
        - Extracción de vegetación usando el Índice de Exceso de Verde (ExG) con umbral automático (Otsu) o manual
        - Múltiples modos de cálculo:
          - Valores brutos: Calculados de la imagen original
          - Valores enmascarados: Reduce la influencia del suelo y agua de fondo al enfocarse solo en áreas de vegetación
        - Detección de bordes y cálculo de PAR para evaluar la delgadez y el tamaño de las hojas
        - Soporte para procesamiento por lotes de múltiples imágenes
        - Resultados detallados con exportación a CSV
        
        **Cómo Usar:**
        1. **Configuración de Análisis (Barra Lateral)**
           - Seleccione el método de umbral para ExG:
             - Método de Otsu para determinación automática
             - Ajuste manual del umbral para control preciso
           - Elija los índices de vegetación a calcular
        
        2. **Análisis de Imagen Individual**
           - Suba una imagen
           - Vea la imagen original, máscara de vegetación y resultados de detección de bordes
           - Observe la cobertura vegetal y PAR (PAR más alto indica hojas más delgadas o pequeñas)
           - Compare índices entre imagen bruta y áreas enmascaradas
        
        3. **Procesamiento por Lotes**
           - Suba múltiples imágenes
           - Inicie el procesamiento para analizar todas las imágenes
           - Descargue resultados completos en CSV
        
        **Consejos:**
        - Para mejores resultados, use imágenes claras con buen contraste
        - Use primero el método de Otsu, luego cambie al umbral manual si es necesario
        - Valores PAR más altos indican hojas más delgadas o más pequeñas
        - Los valores enmascarados son útiles cuando el fondo (suelo, agua) podría afectar el análisis
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
        - Excess Green Index (ExG)を用いた植生抽出（大津の方法による自動閾値設定または手動設定）
        - 複数の計算モード：
          - 元画像の値：画像全体から計算
          - マスク値：植生領域のみを対象とし、背景の土壌や水の影響を低減
        - エッジ検出と周長面積比（PAR）による葉の細さや小ささの評価
        - 複数画像の一括処理機能
        - 詳細な解析結果のCSVエクスポート

        **使用方法：**
        1. **解析設定（サイドバー）**
           - ExGの閾値設定方法を選択：
             - 大津の方法による自動閾値決定
             - 手動での閾値調整
           - 計算する植生指数を選択

        2. **単一画像解析**
           - 画像をアップロード
           - 元画像、植生マスク、エッジ検出結果を確認
           - 植生被覆率とPAR（高いPARは葉が細いまたは小さいことを示す）を確認
           - 元画像とマスク領域での指数値を比較

        3. **バッチ処理**
           - 複数の画像をアップロード
           - 処理を開始して全画像を解析
           - 結果をCSVでダウンロード

        **ヒント：**
        - より良い結果を得るために、コントラストの良い鮮明な画像を使用してください
        - まず大津の方法を試し、必要に応じて手動閾値に切り替えてください
        - PAR値が高いほど、葉が細いまたは小さいことを示します
        - 背景（土壌や水）が解析に影響を与える可能性がある場合、マスク値が有用です
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
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, Dict], Dict[str, float]]:
    # リサイズ
    image = resize_if_needed(image)
    
    # float32で計算
    image_float = image.astype(np.float32) 
    b, g, r = cv2.split(image_float)
    del image_float
    gc.collect()
    
    # 元画像の合計値を保存
    total_raw = r + g + b
    
    # Raw mode用の正規化
    nr_raw = np.divide(r, total_raw, out=np.zeros_like(r), where=total_raw!=0)
    ng_raw = np.divide(g, total_raw, out=np.zeros_like(g), where=total_raw!=0)
    nb_raw = np.divide(b, total_raw, out=np.zeros_like(b), where=total_raw!=0)
    
    # マスク生成
    exg = 2 * ng_raw - nr_raw - nb_raw
    if threshold_method == "otsu":
        exg_uint8 = ((exg + 1) * 127.5).astype(np.uint8)
        thresh, _ = cv2.threshold(exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = (thresh / 127.5) - 1
        del exg_uint8
    else:
        threshold = exg_threshold
    
    binary_mask = (exg >= threshold).astype(np.uint8) * 255
    edges = cv2.Canny(binary_mask, 100, 200)
    
    # マスク処理
    mask_bool = binary_mask > 0
    
    # マスクされた値の計算
    r_masked = r * mask_bool
    g_masked = g * mask_bool
    b_masked = b * mask_bool
    
    # マスク領域用の正規化値
    nr_masked = np.divide(r_masked, total_raw, out=np.zeros_like(r), where=total_raw!=0)
    ng_masked = np.divide(g_masked, total_raw, out=np.zeros_like(g), where=total_raw!=0)
    nb_masked = np.divide(b_masked, total_raw, out=np.zeros_like(b), where=total_raw!=0)
    
    # マスク画像の作成
    masked_image = image.copy()
    masked_image[binary_mask == 0] = 0
    
    images = {
        "binary": binary_mask,
        "masked": masked_image,
        "edges": edges
    }
    
    # ピクセル数の計算
    pixels = {
        "veg": np.count_nonzero(binary_mask),
        "total": binary_mask.size,
        "perimeter": np.count_nonzero(edges)
    }
    
    # 指標計算
    indices_result = {"raw": {}, "masked": {}}
    for index_name in selected_indices:
        if index_name == "INT":
            raw_value = ALGORITHMS[index_name][1](r, g, b)
            masked_value = ALGORITHMS[index_name][1](r_masked, g_masked, b_masked)
        else:
            raw_value = ALGORITHMS[index_name][1](nr_raw, ng_raw, nb_raw)
            masked_value = ALGORITHMS[index_name][1](nr_masked, ng_masked, nb_masked)
        
        indices_result["raw"][index_name] = float(np.mean(raw_value))
        indices_result["masked"][index_name] = float(np.mean(raw_value[mask_bool])) if pixels["veg"] > 0 else 0.0
        del raw_value, masked_value
    
    # PAR計算
    par = {
        "value": pixels["perimeter"] / pixels["veg"] if pixels["veg"] > 0 else 0
    }
    
    return images, pixels, indices_result, par

def process_batch(uploaded_files, threshold_method, exg_threshold, selected_indices, lang, progress_bar=None, status_text=None):
    results = []
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files, 1):
        try:
            if progress_bar is not None:
                progress = i / total_files
                progress_bar.progress(progress)
                status_text.text(f"{get_text('processing', lang)} {file.name} ({i}/{total_files})")
            
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            images, pixels, indices, par = process_single_image(
                image, threshold_method, exg_threshold, selected_indices
            )
            
            # 基本情報
            result = {
                get_text("file_name", lang): file.name,
                get_text("coverage_rate_percent", lang): (pixels['veg'] / pixels['total']) * 100,
                get_text("veg_pixels", lang): pixels['veg'],
                get_text("total_pixels", lang): pixels['total'],
                "PAR": par['value'],
                get_text("threshold_method_col", lang): get_text("otsu_method" if threshold_method == "otsu" else "exg_threshold_method", lang),
                get_text("threshold_value", lang): get_text("automatic", lang) if threshold_method == "otsu" else str(exg_threshold)
            }
            
            # 指標値
            for key in selected_indices:
                result[f'{ALGORITHMS[key][0]} (Raw)'] = indices['raw'][key]
                result[f'{ALGORITHMS[key][0]} (Masked)'] = indices['masked'][key]
            
            results.append(result)
            
        except Exception as e:
            st.error(f"{get_text('error_occurred', lang)} {str(e)} in file: {file.name}")
            continue
            
    return results

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
            type=["png", "jpg", "jpeg"],
            key="single_image_uploader"
        )
        
        if uploaded_file:
            try:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                images, pixels, indices, par = process_single_image(
                    image, threshold_method, exg_threshold, selected_indices
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(image, caption=get_text("original_image", lang))
                with col2:
                    st.image(images["masked"], caption=get_text("vegetation_result", lang))
                with col3:
                    st.image(images["edges"], caption="Edge Detection Result")
                
                metrics_cols = st.columns(3)
                with metrics_cols[0]:
                    st.metric(get_text("coverage_rate", lang), 
                             f"{(pixels['veg'] / pixels['total']) * 100:.2f}%")
                with metrics_cols[1]:
                    st.metric(get_text("veg_pixels", lang), f"{pixels['veg']:,}")
                with metrics_cols[2]:
                    st.metric("PAR", f"{par['value']:.4f}")
                
                if indices:
                    st.subheader(get_text("index_results", lang))
                    mode_cols = st.columns(2)
                    with mode_cols[0]:
                        st.write("Raw Image Values:")
                        for key in selected_indices:
                            st.write(f"{ALGORITHMS[key][0]}: {indices['raw'][key]:.4f}")
                    with mode_cols[1]:
                        st.write("Masked Area Values:")
                        for key in selected_indices:
                            st.write(f"{ALGORITHMS[key][0]}: {indices['masked'][key]:.4f}")
            
            except Exception as e:
                st.error(f"{get_text('error_occurred', lang)} {str(e)}")

    with tab2:
        uploaded_files = st.file_uploader(
            get_text("upload_multiple", lang),
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="batch_image_uploader"
        )
        
        if uploaded_files and len(selected_indices) > 0:
            if st.button(get_text("start_batch", lang), type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # バッチ処理実行
                try:
                    results = process_batch(uploaded_files, threshold_method, exg_threshold, selected_indices, lang)
                    
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
                except Exception as e:
                    st.error(f"Batch processing failed: {str(e)}")

if __name__ == "__main__":
    main()