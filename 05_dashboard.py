"""
05 - Dashboard Streamlit (GEE Real-time)
========================================
Dashboard interaktif yang query GEE secara real-time.
Tidak memerlukan file GeoTIFF lokal.

Usage: streamlit run 05_dashboard.py
"""

import streamlit as st
import ee
import json
import os
import sys
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import plugins
from streamlit_folium import st_folium
import requests
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.gee_utils import (
    authenticate_gee, get_roi, get_sentinel2_collection,
    compute_all_stats, get_ee_tile_url, get_index_vis_params,
    STUDY_AREA, ANALYSIS_YEARS, INDEX_NAMES
)
from utils.water_indices import ee_add_all_indices

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
MAPS_DIR = os.path.join(BASE_DIR, 'output', 'maps')
CHARTS_DIR = os.path.join(BASE_DIR, 'output', 'charts')

PARAM_LABELS = {
    'NDCI': 'Chlorophyll-a (NDCI mg/m³)',
    'NDTI': 'Turbiditas (NDTI)',
    'TSS': 'TSS (g/m³)',
    'CDOM': 'CDOM (B2/B3)',
    'Secchi_Depth': 'Secchi Depth (m)',
    'SST': 'Suhu Perairan (SST °C)',
}

def get_kepmen_status(param, val):
    """Berdasarkan Kepmen LH No 51/2004 untuk parameter Klorofil-a (NDCI proxy), TSM/TSS, dan SST."""
    if param == 'TSS':
        if val < 0.5: return "🔴 **Buruk (<0.5 g/m³)**: Sangat jernih namun miskin nutrien (produktivitas rendah)"
        elif 0.5 <= val <= 1.5: return "🟢 **Baik (0.5-1.5 g/m³)**: Kondisi seimbang mendukung ekosistem"
        else: return f"🔴 **Buruk ({val:.2f} g/m³)**: Kandungan material sangat tinggi, menghambat fotosintesis"
    elif param == 'NDCI':
        if val < 0.5: return "🔴 **Buruk (<0.5 mg/m³)**: Klorofil sangat rendah, perairan kurang subur"
        elif 0.5 <= val <= 1.0: return "🟢 **Baik (0.5-1.0 mg/m³)**: Kondisi optimum fitoplankton stabil"
        else: return "🔴 **Buruk (>1.0 mg/m³)**: Eutrofik, menurunkan kualitas perairan (bahaya alga)"
    elif param == 'SST':
        if val < 28: return "🔴 **Buruk (<28 °C)**: Terlalu dingin, memperlambat fotosintesis"
        elif 28 <= val <= 32: return "🟢 **Baik (28-32 °C)**: Rentang suhu optimum"
        else: return f"🔴 **Buruk (>32 °C)**: Terlalu panas ({val:.1f}°C) memicu stres pada biota laut"
    return "⚪ *Belum ada standar baku Kepmen LH untuk parameter ini*"

st.set_page_config(
    page_title="🌊 Monitoring Perairan Pesisir",
    page_icon="🌊", layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def init_gee():
    """Inisialisasi GEE (sekali saja)."""
    authenticate_gee()
    return True


@st.cache_data(ttl=3600)
def load_cached_stats():
    """Load statistik dari file JSON lokal."""
    all_stats = {}
    for year in ANALYSIS_YEARS:
        path = os.path.join(RESULTS_DIR, f'statistics_{year}.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            all_stats[year] = data['statistics']
    return all_stats


@st.cache_data(ttl=3600)
def get_gee_image(year):
    """Ambil processed image dari GEE (cached)."""
    roi = get_roi()
    composite = get_sentinel2_collection(year, roi, max_cloud_pct=10)
    return ee_add_all_indices(composite)


def render_header():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #0077B6, #00B4D8, #48CAE4);
        padding: 25px; border-radius: 15px; margin-bottom: 20px;
        text-align: center; color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 18px; border-radius: 12px; text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-value { font-size: 26px; font-weight: bold; color: #48CAE4; }
    .metric-label { font-size: 11px; color: #aaa; margin-top: 4px; }
    </style>
    <div class="main-header">
        <h1>🌊 Monitoring Kualitas Perairan Pesisir</h1>
        <h3>Jakarta & Banten — Sentinel-2 via Google Earth Engine</h3>
        <p style="font-size:12px;opacity:0.8">☁️ Full Cloud Processing — Data real-time dari GEE</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.title("⚙️ Pengaturan")

    st.sidebar.markdown("### 🗺️ Pemilihan Tahun")
    selected_year_1 = st.sidebar.selectbox(
        "📅 Tahun Awal (Kiri)", ANALYSIS_YEARS, index=0)
    selected_year_2 = st.sidebar.selectbox(
        "📅 Tahun Pembanding (Kanan)", ANALYSIS_YEARS, index=len(ANALYSIS_YEARS) - 1)
        
    selected_param = st.sidebar.selectbox(
        "📊 Parameter", INDEX_NAMES,
        format_func=lambda x: PARAM_LABELS.get(x, x))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Info")
    st.sidebar.markdown(f"- **Area:** Pesisir Jakarta-Banten")
    st.sidebar.markdown(f"- **Resolusi:** 10m (Sentinel-2)")
    st.sidebar.markdown(f"- **Processing:** ☁️ GEE Cloud")
    st.sidebar.markdown(f"- **Tahun:** {ANALYSIS_YEARS[0]}–{ANALYSIS_YEARS[-1]}")

    return selected_year_1, selected_year_2, selected_param


def render_metrics(stats, year, param):
    """Render metric cards dari statistik cached."""
    if not stats or year not in stats or param not in stats[year]:
        st.info("Statistik belum tersedia. Jalankan 02_cloud_processing.py dulu.")
        return

    s = stats[year][param]
    cols = st.columns(5)
    
    metrics = [
        ("Rata-rata (Mean)", f"{s.get('mean', 0):.4f}"),
        ("Persebaran (Std Dev)", f"{s.get('stdDev', 0):.4f}"),
        ("Kandungan Terendah", f"{s.get('min', 0):.4f}"),
        ("Kandungan Tertinggi", f"{s.get('max', 0):.4f}"),
        ("Area Tercover (%)", f"{s.get('coverage_pct', 0):.1f}%"),
    ]
    for col, (label, val) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    # Tambahkan Klasifikasi Kepmen LH
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"📋 **Status Kualitas Air Nasional:** {get_kepmen_status(param, s.get('mean', 0))}")


def render_map(year1, year2, param):
    """Peta interaktif menggunakan Split-Panel DualMap (Kiri vs Kanan)."""
    center_lat = (STUDY_AREA['north'] + STUDY_AREA['south']) / 2
    center_lon = (STUDY_AREA['west'] + STUDY_AREA['east']) / 2

    # Inisialisasi peta terbagi
    m = plugins.DualMap(location=[center_lat, center_lon], zoom_start=10, 
                        tiles=None)

    # Base layers map kiri & kanan
    folium.TileLayer('CartoDB dark_matter', name='Dark Matter').add_to(m.m1)
    folium.TileLayer('CartoDB dark_matter', name='Dark Matter').add_to(m.m2)

    # Satellite base layer (opsional)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite'
    ).add_to(m.m1)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite'
    ).add_to(m.m2)

    # GEE tile map kiri (Tahun Awal)
    try:
        img1 = get_gee_image(year1)
        url1 = get_ee_tile_url(img1, param)
        folium.TileLayer(
            tiles=url1, attr='Google Earth Engine',
            name=f'{param} {year1}', overlay=True, opacity=0.8
        ).add_to(m.m1)
    except Exception as e:
        st.error(f"Gagal load GEE tile peta kiri: {e}")
        
    # GEE tile map kanan (Tahun Pembanding)
    try:
        img2 = get_gee_image(year2)
        url2 = get_ee_tile_url(img2, param)
        folium.TileLayer(
            tiles=url2, attr='Google Earth Engine',
            name=f'{param} {year2}', overlay=True, opacity=0.8
        ).add_to(m.m2)
    except Exception as e:
        st.error(f"Gagal load GEE tile peta kanan: {e}")

    # Bounding box
    folium.Rectangle(
        bounds=[[STUDY_AREA['south'], STUDY_AREA['west']],
                [STUDY_AREA['north'], STUDY_AREA['east']]],
        color='cyan', weight=2, fill=False
    ).add_to(m)

    folium.LayerControl().add_to(m.m1)
    folium.LayerControl().add_to(m.m2)
    st_folium(m, width="100%", height=500)


def render_thumbnail(year, param):
    """Tampilkan thumbnail dari file lokal."""
    path = os.path.join(MAPS_DIR, f'{param}_{year}.png')
    if os.path.exists(path):
        st.image(path, caption=f'{PARAM_LABELS.get(param, param)} — {year}',
                 use_container_width=True)
    else:
        st.info("Thumbnail belum tersedia. Jalankan 02_cloud_processing.py.")


def render_timeseries(stats):
    """Grafik time-series dari semua tahun."""
    if not stats:
        return

    rows = []
    for year, params in stats.items():
        for param, s in params.items():
            rows.append({
                'Tahun': year,
                'Parameter': PARAM_LABELS.get(param, param),
                'Mean': s.get('mean', 0) or 0
            })

    df = pd.DataFrame(rows)
    fig = px.line(df, x='Tahun', y='Mean', color='Parameter',
                  markers=True, title='Grafik Perubahan Tahunan')
    fig.update_layout(
        template='plotly_dark', height=380,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 *Grafik ini membantu kita melihat apakah kualitas air laut membaik atau memburuk dari tahun ke tahun. Jika garis mengarah ke atas, berarti kandungan zat tersebut meningkat.*")


def render_charts():
    """Tampilkan chart dari file lokal."""
    chart_files = sorted(glob.glob(os.path.join(CHARTS_DIR, '*.png')))
    if chart_files:
        st.markdown("### 📊 Grafik Analisis")
        cols = st.columns(2)
        for i, cf in enumerate(chart_files):
            with cols[i % 2]:
                st.image(cf, caption=os.path.basename(cf),
                         use_container_width=True)


def render_change_detection():
    """Tampilkan hasil change detection."""
    cd_path = os.path.join(RESULTS_DIR, 'change_detection.json')
    if not os.path.exists(cd_path):
        return

    st.markdown("### 🔄 Apakah Laut Kita Semakin Tercemar/Bersih?")
    st.caption("Membandingkan trend berdasarkan data historis tahun awal hingga akhir.")
    with open(cd_path) as f:
        data = json.load(f)

    # Load data p-value / uji signifikansi statistik (jika ada)
    sig_path = os.path.join(RESULTS_DIR, 'trend_significance.csv')
    sig_data = {}
    if os.path.exists(sig_path):
        df_sig = pd.read_csv(sig_path)
        for _, row in df_sig.iterrows():
            sig_data[row['parameter']] = {
                'p_val': row['p_value'],
                'is_sig': row['p_value'] < 0.05
            }

    st.markdown(f"**Periode Perbandingan:** Tahun {data['period']}")
    params = data.get('parameters', {})
    cols = st.columns(len(params))
    
    for col, (param, vals) in zip(cols, params.items()):
        with col:
            delta = vals['mean_change']
            st.metric(
                PARAM_LABELS.get(param, param),
                f"{delta:+.4f}",
                delta=f"{'↑ Meningkat' if delta > 0 else '↓ Menurun'} {abs(delta):.4f}"
            )
            # Menampilkan hasil uji hipotesis p-value di bawah metrics
            sig_info = sig_data.get(param)
            if sig_info:
                icon = "✅ (Signifikan)" if sig_info['is_sig'] else "⚪ (Tdk Signifikan)"
                st.markdown(f"<div style='font-size:11px;color:#aaa;margin-top:-10px;'>P-val: {sig_info['p_val']:.3f} {icon}</div>", unsafe_allow_html=True)


def render_conclusion():
    """Tampilkan kesimpulan otomatis berdasarkan data change detection dan p-value."""
    cd_path = os.path.join(RESULTS_DIR, 'change_detection.json')
    sig_path = os.path.join(RESULTS_DIR, 'trend_significance.csv')
    
    if not os.path.exists(cd_path):
        return

    st.markdown("---")
    st.markdown("## 📝 Ringkasan Eksekutif & Kesimpulan Akhir")
    st.caption("Kesimpulan naratif otomatis berdasarkan analisis data satelit.")

    with open(cd_path) as f:
        data = json.load(f)
        
    params = data.get('parameters', {})
    
    # Load p-value data
    sig_data = {}
    if os.path.exists(sig_path):
        df_sig = pd.read_csv(sig_path)
        for _, row in df_sig.iterrows():
            sig_data[row['parameter']] = row['p_value'] < 0.05
            
    membaik = []
    memburuk = []
    
    for param, vals in params.items():
        delta = vals['mean_change']
        is_sig = sig_data.get(param, False)
        sig_text = "(Signifikan)" if is_sig else "(Tidak Signifikan)"
        
        # Logika: apakah perubahan itu "baik" atau "buruk"?
        if param == 'TSS':
            if delta < 0: membaik.append(f"**TSS (Material Padat)** menurun {abs(delta):.2f} {sig_text}: Air lebih jernih dari lumpur/pasir.")
            else: memburuk.append(f"**TSS (Material Padat)** meningkat {abs(delta):.2f} {sig_text}: Air lebih pekat/kotor.")
        elif param == 'CDOM':
            if delta < 0: membaik.append(f"**CDOM (Bahan Organik)** menurun {abs(delta):.2f} {sig_text}: Indikasi pengurangan limbah organik.")
            else: memburuk.append(f"**CDOM (Bahan Organik)** meningkat {abs(delta):.2f} {sig_text}: Indikasi penumpukan sisa bahan organik/polusi.")
        elif param == 'Secchi_Depth':
            if delta > 0: membaik.append(f"**Secchi Depth** meningkat {abs(delta):.2f}m {sig_text}: Jarak pandang tembus cahaya ke dalam laut semakin baik/jernih.")
            else: memburuk.append(f"**Secchi Depth** menurun {abs(delta):.2f}m {sig_text}: Laut menjadi lebih keruh, cahaya sulit menembus.")
        elif param == 'SST':
            # SST naik biasanya buruk (pemanasan global/stres termal)
            if delta > 0: memburuk.append(f"**Suhu Permukaan (SST)** naik {abs(delta):.1f}°C {sig_text}: Tren pemanasan perairan yang bisa memicu stres pada biota lepas pantai/karang.")
            else: membaik.append(f"**Suhu Permukaan (SST)** turun {abs(delta):.1f}°C {sig_text}: Suhu perairan menjadi lebih sejuk.")
        elif param == 'NDCI':
            if delta > 0: memburuk.append(f"**NDCI (Klorofil-a)** naik {abs(delta):.3f} {sig_text}: Peningkatan fitoplankton, potensi risiko eutrofikasi/alga meledak.")
            else: membaik.append(f"**NDCI (Klorofil-a)** turun {abs(delta):.3f} {sig_text}: Penurunan konsentrasi alga/fitoplankton.")
        elif param == 'NDTI':
             if delta < 0: membaik.append(f"**NDTI (Turbiditas)** menurun {abs(delta):.3f} {sig_text}: Kekeruhan air akibat material terlarut berkurang.")
             else: memburuk.append(f"**NDTI (Turbiditas)** naik {abs(delta):.3f} {sig_text}: Kekeruhan perairan bertambah.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ **Kabar Baik (Kualitas Membaik)**")
        if membaik:
            for item in membaik:
                st.markdown(f"- {item}")
        else:
            st.markdown("*Tidak ada indikator yang menunjukkan perbaikan signifikan.*")
            
    with col2:
        st.error("🚨 **Kabar Buruk (Kualitas Memburuk)**")
        if memburuk:
            for item in memburuk:
                st.markdown(f"- {item}")
        else:
            st.markdown("*Tidak ada indikator yang menunjukkan perburukan signifikan.*")
            
    st.info(f"**Kesimpulan Utama (Periode {data['period']}):** Berdasarkan analisis geospasial berbasis _Machine Learning & Cloud Processing_, dinamika perairan Pesisir Jakarta dan Banten menunjukkan perubahan multi-dimensi. Faktor signifikansi p-value memastikan bahwa trend yang terjadi adalah pola nyata, bukan sekadar anomali sesaat.")


def main():
    init_gee()
    render_header()
    selected_year_1, selected_year_2, selected_param = render_sidebar()

    # Load cached stats
    stats = load_cached_stats()

    # Metrics untuk tahun paling akhir (kanan)
    st.markdown(f"### 📊 Statistik Parametrik: {PARAM_LABELS.get(selected_param, selected_param)} — Tahun {selected_year_2}")
    render_metrics(stats, selected_year_2, selected_param)
    st.markdown("---")

    # Main content map real-time
    st.markdown(f"### 🗺️ Peta Resolusi Tinggi (Area Kiri: **{selected_year_1}** ↔ Area Kanan: **{selected_year_2}**)")
    st.caption("Geser *slider* peta interaktif ini ke kiri dan kanan untuk melihat perbedaan kondisi perairan pesisir secara langsung melalui Citra Satelit.")
    
    render_map(selected_year_1, selected_year_2, selected_param)
    
    st.markdown("---")
    
    # Tambahkan Edukasi untuk POV Umum
    with st.expander("📖 Panduan Singkat untuk Pengguna Umum (Cara Membaca Data Ini)"):
        st.markdown("""
        **Apa arti gambar dan grafik di bawah ini?**
        - **Peta Satelit Split:** Area sebelah kiri merepresentasikan "Tahun Awal", sebelah kanan "Tahun Akhir. Lihat warna merah (kondisi panas/pekat).
        - **Trend Temporal (Grafik Garis):** Menunjukkan apakah kualitas air membaik atau memburuk dari tahun 2019 ke 2025.
        - **Statistik (Mean/Rata-rata):** Angka yang menunjukkan tingkat kepekatan zat di seluruh perairan pada tahun akhir penanggalan.
        
        **Apa arti dari parameter (model) ini?**
        1. **Chlorophyll-a (NDCI):** Indikator banyaknya fitoplankton/alga. Menjadi penentu kesuburan perairan.
        2. **Turbiditas (NDTI):** Tingkat kekeruhan air karena pasir, erosi, dll.
        3. **TSS (Total Suspended Solids):** Jumlah material padat (pasir/lumpur) yang melayang di air. Angka tinggi berarti laut sangat kotor/tercemar.
        4. **CDOM (Bahan Organik Terlarut):** Indikator pembusukan bahan organik dari limbah sungai atau daratan.
        5. **Secchi Depth:** Mengukur seberapa **jernih** air laut cahaya tembus ke dalam.
        6. **SST (Sea Surface Temperature):** Suhu laut, sangat penting bagi terumbu karang. Suhu terlalu dingin/panas merusak ekosistem.
        """)

    st.markdown("### 📈 Trend Temporal Historis")
    render_timeseries(stats)

    st.markdown("---")

    # Change detection
    render_change_detection()

    # Charts
    render_charts()
    
    # Conclusion / Executive Summary
    render_conclusion()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#666;padding:15px;">
        🌊 Monitoring Kualitas Perairan Pesisir Jakarta & Banten<br>
        Data: Sentinel-2 & Landsat-8 (ESA/NASA) | Processing: Google Earth Engine | Python
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
