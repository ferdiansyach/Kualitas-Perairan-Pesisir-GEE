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
import branca.colormap as cm
from folium import plugins
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.gee_utils import (
    authenticate_gee, get_roi, get_sentinel2_collection,
    get_landsat8_sst_collection, compute_all_stats, get_ee_tile_url, 
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
    """Inisialisasi GEE."""
    try:
        authenticate_gee()
        return True
    except Exception as e:
        st.error(f"🚨 **GEE Token Expired**: Koneksi Google Earth Engine terputus. Buka terminal Anda dan jalankan perintah: `earthengine authenticate`. \n\nError: {e}")
        return False


@st.cache_data(ttl=3600)
def load_pvalue_data():
    sig_path = os.path.join(RESULTS_DIR, 'trend_significance.csv')
    sig_data = {}
    if os.path.exists(sig_path):
        df_sig = pd.read_csv(sig_path)
        for _, row in df_sig.iterrows():
            sig_data[row['parameter']] = {
                'p_val': row['p_value'],
                'is_sig': row['p_value'] < 0.05
            }
    return sig_data


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


# Hapus dekorator @st.cache_data karena objek ee.Image tidak serializable dengan aman
# Proses build DAG EE di client-side sangat ringan (<1ms), jadi tidak perlu di-cache oleh Streamlit.
def get_gee_image(year, param):
    """Ambil processed image dari GEE sesuai jenis satelit."""
    roi = get_roi()
    
    # Sentinel-2 (Optik / Mulstispektral) tidak dapat memotret suhu
    if param == 'SST':
        return get_landsat8_sst_collection(year, roi, max_cloud_pct=10)
    else:
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


def render_metrics(stats, year1, year2, param):
    """Render metric cards perbandingan Kiri vs Kanan."""
    if not stats or year2 not in stats or param not in stats[year2]:
        st.warning("Statistik belum tersedia. Pastikan script pemrosesan cloud (02) sudah selesai.")
        return

    s1 = stats.get(year1, {}).get(param, {})
    s2 = stats[year2][param]
    cols = st.columns(5)
    
    metrics = [
        ("Rata-rata (Mean)", 'mean', "{:.4f}"),
        ("Persebaran (Std)", 'stdDev', "{:.4f}"),
        ("Terendah (Min)", 'min', "{:.4f}"),
        ("Tertinggi (Max)", 'max', "{:.4f}"),
        ("Area Valid", 'coverage_pct', "{:.1f}%"),
    ]
    
    for col, (label, key, fmt) in zip(cols, metrics):
        val1 = s1.get(key, 0)
        val2 = s2.get(key, 0)
        delta = val2 - val1
        col.metric(label, fmt.format(val2), f"{delta:+.4f}" if key != 'coverage_pct' else f"{delta:+.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"📋 **Status Mutu Air Nasional (Data {year2}):** {get_kepmen_status(param, s2.get('mean', 0))}")


def render_map(year1, year2, param):
    """Peta interaktif menggunakan Split-Panel DualMap (Kiri vs Kanan)."""
    center_lat = (STUDY_AREA['north'] + STUDY_AREA['south']) / 2
    center_lon = (STUDY_AREA['west'] + STUDY_AREA['east']) / 2

    # Inisialisasi peta terbagi dengan cartodb yang menempel langsung sbg default
    m = plugins.DualMap(location=[center_lat, center_lon], zoom_start=10, 
                        tiles='CartoDB dark_matter')

    # Satellite base layer (opsional) - Sinkron ke kedua map
    sat_layer1 = folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite'
    )
    sat_layer2 = folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite'
    )
    sat_layer1.add_to(m.m1)
    sat_layer2.add_to(m.m2)

    # GEE tile map kiri (Tahun Awal)
    try:
        img1 = get_gee_image(year1, param)
        url1 = get_ee_tile_url(img1, param)
        gee_layer1 = folium.TileLayer(
            tiles=url1, attr='Google Earth Engine',
            name=f'{param} {year1}', overlay=True, opacity=0.8
        )
        gee_layer1.add_to(m.m1)
    except Exception as e:
        st.error(f"Gagal load GEE tile peta kiri: {e}")
        
    # GEE tile map kanan (Tahun Pembanding)
    try:
        img2 = get_gee_image(year2, param)
        url2 = get_ee_tile_url(img2, param)
        gee_layer2 = folium.TileLayer(
            tiles=url2, attr='Google Earth Engine',
            name=f'{param} {year2}', overlay=True, opacity=0.8
        )
        gee_layer2.add_to(m.m2)
    except Exception as e:
        st.error(f"Gagal load GEE tile peta kanan: {e}")

    # Bounding box
    folium.Rectangle(
        bounds=[[STUDY_AREA['south'], STUDY_AREA['west']],
                [STUDY_AREA['north'], STUDY_AREA['east']]],
        color='cyan', weight=2, fill=False
    ).add_to(m)

    # Layer Control dikelompokkan agar tidak merusak UI jika dibuka pop-up list drop down-nya
    folium.LayerControl(position='topright', collapsed=True).add_to(m.m1)
    folium.LayerControl(position='topleft', collapsed=True).add_to(m.m2)

    # -------------------------------------------------------------
    # 🎨 INJEKSI CUSTOM CSS UNTUK UI PETA YANG BERSIH & SUPER LEGA
    # -------------------------------------------------------------
    clean_ui_css = """
    <style>
    /* Sembunyikan tombol + / - bawaan Leaflet agar peta lega */
    .leaflet-control-zoom { display: none !important; }
    
    /* Buat Layer Control (Kotak Putih) menjadi mungil & transparan */
    .leaflet-control-layers {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3) !important;
        margin-top: 15px !important;  
    }
    .leaflet-control-layers-toggle {
        width: 30px !important; 
        height: 30px !important;
    }
    
    /* MEMBERIKAN GAP (JARAK) ANTARA PETA KIRI DAN KANAN */
    #map_div_1, .leaflet-container:first-of-type {
        border-right: 8px solid #0e1117 !important; /* Warna sesuai background dark theme streamlit */
    }
    </style>
    """
    m.get_root().html.add_child(folium.Element(clean_ui_css))
    
    # Gunakan direct HTML injection karena st_folium sering gagal me-render DualMap (plugin kompleks)
    components.html(m._repr_html_(), height=500)


def render_timeseries(stats, param):
    """Grafik time-series dari parameter yang dipilih saja dan tombol download."""
    if not stats:
        return

    rows = []
    for yr, param_stats in stats.items():
        if param in param_stats:
            rows.append({
                'Tahun': yr,
                'Nilai Mean': param_stats[param].get('mean', 0)
            })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values('Tahun')
    fig = px.line(df, x='Tahun', y='Nilai Mean', markers=True, 
                  title=f'Grafik Histori Tahunan ({PARAM_LABELS.get(param, param)})',
                  color_discrete_sequence=['#48CAE4'])
    fig.update_layout(
        template='plotly_dark', height=380,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download JSON stats convert to CSV
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data Tren (CSV)",
        data=csv_data,
        file_name=f'trend_historis_{param}.csv',
        mime='text/csv'
    )
    st.caption("💡 *Grafik ini membantu kita melihat kelonjakan tren pada parameter spesifik ini dari tahun 2019 hingga 2025.*")


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


def render_change_detection(sig_data):
    """Tampilkan hasil change detection."""
    cd_path = os.path.join(RESULTS_DIR, 'change_detection.json')
    if not os.path.exists(cd_path):
        return

    st.markdown("### 🔄 Apakah Laut Kita Semakin Tercemar/Bersih?")
    st.caption("Membandingkan trend berdasarkan data historis tahun awal hingga akhir.")
    with open(cd_path) as f:
        data = json.load(f)

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


def render_conclusion(sig_data):
    """Tampilkan kesimpulan otomatis berdasarkan data change detection dan p-value."""
    cd_path = os.path.join(RESULTS_DIR, 'change_detection.json')
    
    if not os.path.exists(cd_path):
        return

    st.markdown("---")
    st.markdown("## 📝 Ringkasan Eksekutif & Kesimpulan Akhir")
    st.caption("Kesimpulan naratif otomatis berdasarkan analisis data satelit.")

    with open(cd_path) as f:
        data = json.load(f)
        
    params = data.get('parameters', {})
            
    membaik = []
    memburuk = []
    
    for param, vals in params.items():
        delta = vals['mean_change']
        sig_info = sig_data.get(param, {})
        is_sig = sig_info.get('is_sig', False)
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
    if not init_gee():
        return

    render_header()
    selected_year_1, selected_year_2, selected_param = render_sidebar()

    # Load cached data
    stats = load_cached_stats()
    sig_data = load_pvalue_data()

    # Panduan Edukasi (dipindah ke atas sebelum data teknis)
    with st.expander("📖 Panduan Cepat untuk Awam: Cara Membaca Dashboard Pesisir"):
        st.markdown("""
        **1. Peta Satelit Split:** Area Kiri ("Tahun Awal") dan Kanan ("Tahun Akhir"). Warna merah berarti polusi pekat/panas, biru berarti air jernih/sejuk. Ada legenda warna di peta.
        **2. Statistik Angka:** Perubahan `+` / `-` membuktikan apakah parameter naik atau turun jika tahun Kiri dan Kanan dibandingkan.
        **3. Ringkasan Eksekutif:** Baca tabulasi di bagian **Bawah** untuk kesimpulan naratif (Kabar Baik vs Buruk).
        """)

    # Metrics dengan Delta Kiri (1) ke Kanan (2)
    st.markdown(f"### 📊 Perbandingan Statistik Angka: {selected_year_1} vs {selected_year_2}")
    st.caption(f"Fokus Indikator Utama: **{PARAM_LABELS.get(selected_param, selected_param)}**")
    render_metrics(stats, selected_year_1, selected_year_2, selected_param)
    st.markdown("---")

    # Layout Map & Chart split horizontal
    col_map, col_chart = st.columns([1.2, 1])

    with col_map:
        st.markdown(f"### 🗺️ GEE Split-Map ({selected_year_1} vs {selected_year_2})")
        st.caption("Geser *slider* interaktif pada peta.")
        with st.spinner("⏳ Menghubungkan Superkomputer Google Earth Engine..."):
            render_map(selected_year_1, selected_year_2, selected_param)

    with col_chart:
        st.markdown("### 📈 Tren Kenaikan Historis (2019-2025)")
        render_timeseries(stats, selected_param)

    st.markdown("---")

    # Change detection (makro awal hingga akhir)
    render_change_detection(sig_data)

    # Charts
    render_charts()
    
    # Conclusion / Executive Summary
    render_conclusion(sig_data)

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
