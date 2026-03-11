"""
Google Earth Engine Utility Functions (Cloud Processing)
========================================================
Helper functions untuk pemrosesan citra satelit Sentinel-2
sepenuhnya di server Google Earth Engine — tanpa download lokal.
"""

import ee
import os
import json
import requests
import numpy as np
from io import BytesIO
import yaml

_cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(_cfg_path, encoding='utf-8') as _f:
    _CFG = yaml.safe_load(_f)

PROJECT_ID = _CFG['gee']['project_id']
ANALYSIS_YEARS = _CFG['analysis']['years']

_bbox = _CFG['study_area']['bbox']
STUDY_AREA = {
    'west': _bbox[0],
    'south': _bbox[1],
    'east': _bbox[2],
    'north': _bbox[3],
}

S2_BANDS = {
    'B2': 'Blue', 'B3': 'Green', 'B4': 'Red',
    'B5': 'Red_Edge_1', 'B6': 'Red_Edge_2',
    'B7': 'Red_Edge_3', 'B8': 'NIR', 'B11': 'SWIR_1',
}
BAND_LIST = list(S2_BANDS.keys())

# 3 Parameter Utama (Klorofil-a, TSM, Suhu Perairan)
INDEX_NAMES = ['NDCI', 'TSS', 'SST']


def authenticate_gee(project_id=PROJECT_ID):
    """Autentikasi dan inisialisasi Google Earth Engine."""
    try:
        ee.Initialize(project=project_id)
        print(f"✅ GEE terkoneksi (project: {project_id})")
    except Exception:
        print("🔐 Memulai autentikasi GEE...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print(f"✅ Autentikasi berhasil! (project: {project_id})")


def get_roi():
    """Membuat Region of Interest (ROI)."""
    bbox = _CFG['study_area']['bbox']
    return ee.Geometry.Rectangle(bbox)


def mask_s2_clouds(image):
    """Cloud masking Sentinel-2 menggunakan QA60 band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))
    return image.updateMask(mask)


def get_sentinel2_collection(year, roi=None, max_cloud_pct=10):
    """
    Mendapatkan median composite Sentinel-2 untuk satu tahun.
    Returns ee.Image (masih di server GEE).
    """
    if roi is None:
        roi = get_roi()

    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi)
                  .filterDate(f'{year}-01-01', f'{year}-12-31')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
                  .map(mask_s2_clouds)
                  .select(BAND_LIST))

    composite = collection.median().clip(roi).toFloat()
    count = collection.size().getInfo()
    print(f"📸 Tahun {year}: {count} scene Sentinel-2 (cloud < {max_cloud_pct}%)")
    return composite


def mask_l8_clouds(image):
    """Cloud masking Landsat 8 Collection 2."""
    qa = image.select('QA_PIXEL')
    cloud_shadow_bit = 1 << 4
    clouds_bit = 1 << 3
    mask = (qa.bitwiseAnd(cloud_shadow_bit).eq(0)
            .And(qa.bitwiseAnd(clouds_bit).eq(0)))
    return image.updateMask(mask)


def get_landsat8_sst_collection(year, roi=None, max_cloud_pct=10):
    """
    Mendapatkan SST composite dari Landsat 8 Tier 1 L2 untuk satu tahun.
    SST diekstrak dari Band 10 Thermal (ST_B10) & dikonversi ke derajat Celcius.
    Returns: ee.Image (SST band)
    """
    if roi is None:
        roi = get_roi()

    collection = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                  .filterBounds(roi)
                  .filterDate(f'{year}-01-01', f'{year}-12-31')
                  .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_pct))
                  .map(mask_l8_clouds))

    def calc_sst_celsius(img):
        # Scale factor Landsat 8 ST: 0.00341802 * DN + 149.0 (Kelvin)
        # Konversi ke Celcius: - 273.15
        st_kelvin = img.select('ST_B10').multiply(0.00341802).add(149.0)
        sst_c = st_kelvin.subtract(273.15).toFloat().rename('SST')
        return sst_c

    sst_composite = collection.map(calc_sst_celsius).median().clip(roi)
    count = collection.size().getInfo()
    print(f"🌡️  Tahun {year}: {count} scene Landsat-8 (SST)")
    return sst_composite


# ============================================================
# CLOUD PROCESSING — Statistik & Analisis di GEE
# ============================================================

def compute_region_stats(image, band_name, roi=None, scale=30):
    """
    Hitung statistik (mean, std, min, max) untuk satu band di GEE server.
    Mengembalikan dict kecil — tidak ada download raster.
    """
    if roi is None:
        roi = get_roi()

    band = image.select(band_name)

    stats = band.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.min(), sharedInputs=True)
            .combine(ee.Reducer.max(), sharedInputs=True)
            .combine(ee.Reducer.count(), sharedInputs=True),
        geometry=roi,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True
    ).getInfo()

    # Parse keys — GEE menambahkan suffix _mean, _stdDev, etc.
    result = {}
    for key, val in stats.items():
        clean = key.replace(f'{band_name}_', '')
        result[clean] = val if val is not None else 0

    # Hitung coverage percentage
    total_pixels = image.select(band_name).unmask(0).reduceRegion(
        reducer=ee.Reducer.count(), geometry=roi, scale=scale,
        maxPixels=1e9, bestEffort=True
    ).getInfo()
    total_key = list(total_pixels.keys())[0]
    total = total_pixels[total_key] or 1
    valid = result.get('count', 0)
    result['coverage_pct'] = round((valid / total) * 100, 2) if total > 0 else 0

    return result


def compute_all_stats(image, index_names=None, roi=None, scale=30):
    """Hitung statistik untuk semua parameter kualitas air."""
    if index_names is None:
        index_names = INDEX_NAMES
    if roi is None:
        roi = get_roi()

    all_stats = {}
    for name in index_names:
        try:
            stats = compute_region_stats(image, name, roi, scale)
            all_stats[name] = stats
            print(f"   📊 {name}: mean={stats.get('mean', 0):.4f}, "
                  f"std={stats.get('stdDev', 0):.4f}")
        except Exception as e:
            print(f"   ⚠️  {name}: gagal — {e}")
            all_stats[name] = {
                'mean': None,
                'stdDev': None,
                'min': None,
                'max': None,
                'coverage_pct': None,
            }

    return all_stats


def save_stats_json(stats, year, output_dir):
    """Simpan statistik sebagai JSON file kecil."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'statistics_{year}.json')
    data = {'year': year, 'statistics': stats}
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"   💾 Disimpan: {filepath}")
    return filepath


# ============================================================
# THUMBNAIL & VISUALISASI dari GEE
# ============================================================

def get_index_vis_params(index_name):
    """Parameter visualisasi untuk setiap indeks."""
    params = {
        'NDCI': {'min': -0.3, 'max': 0.3,
                 'palette': ['blue', 'cyan', 'green', 'yellow', 'red']},
        'NDTI': {'min': -0.3, 'max': 0.3,
                 'palette': ['darkblue', 'blue', 'cyan', 'yellow', 'orange']},
        'TSS':  {'min': 0, 'max': 50,
                 'palette': ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020']},
        'CDOM': {'min': 0.5, 'max': 1.5,
                 'palette': ['yellow', 'orange', 'brown', 'darkred']},
        'Secchi_Depth': {'min': 0.5, 'max': 2.0,
                         'palette': ['red', 'orange', 'yellow', 'cyan', 'blue']},
        'NDWI': {'min': -0.5, 'max': 0.5,
                 'palette': ['brown', 'white', 'blue']},
        'SST':  {'min': 28, 'max': 33,
                 'palette': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']},
    }
    return params.get(index_name, {'min': -1, 'max': 1, 'palette': ['blue', 'white', 'red']})


def get_thumbnail(image, band_name, roi=None, dimensions=512):
    """
    Generate thumbnail PNG dari GEE — download file kecil (~100KB).
    Returns: bytes (PNG image data)
    """
    if roi is None:
        roi = get_roi()

    vis = get_index_vis_params(band_name)
    vis['region'] = roi
    vis['dimensions'] = dimensions
    vis['format'] = 'png'

    url = image.select(band_name).getThumbURL(vis)
    response = requests.get(url, timeout=60)
    return response.content


def save_thumbnail(image, band_name, output_path, roi=None, dimensions=512):
    """Download dan simpan thumbnail sebagai PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    png_data = get_thumbnail(image, band_name, roi, dimensions)
    with open(output_path, 'wb') as f:
        f.write(png_data)
    size_kb = len(png_data) / 1024
    print(f"   🗺️  {band_name} → {output_path} ({size_kb:.0f} KB)")


def generate_all_thumbnails(image, year, output_dir, index_names=None, dimensions=512):
    """Generate semua thumbnail peta untuk satu tahun."""
    if index_names is None:
        index_names = INDEX_NAMES
    os.makedirs(output_dir, exist_ok=True)

    for name in index_names:
        try:
            path = os.path.join(output_dir, f'{name}_{year}.png')
            save_thumbnail(image, name, path, dimensions=dimensions)
        except Exception as e:
            print(f"   ⚠️  Thumbnail {name}: gagal — {e}")


# ============================================================
# SAMPLING — Ambil Data untuk ML
# ============================================================

def sample_training_data(image, n_points=10000, scale=30, roi=None, seed=42):
    """
    Ambil sample points dari GEE untuk training ML lokal.
    Returns: pandas DataFrame (kecil, ~1-5 MB)
    """
    import pandas as pd

    if roi is None:
        roi = get_roi()

    # Sample random points
    samples = image.sample(
        region=roi,
        scale=scale,
        numPixels=n_points,
        seed=seed,
        geometries=False
    )

    # Convert ke list of dicts
    data = samples.getInfo()
    features = data['features']

    rows = [f['properties'] for f in features]
    df = pd.DataFrame(rows)
    print(f"   📋 {len(df)} sample points diambil ({len(df.columns)} fitur)")
    return df


# ============================================================
# TILE LAYER — untuk Dashboard Streamlit
# ============================================================

def get_ee_tile_url(image, band_name):
    """
    Generate tile URL dari GEE untuk digunakan di Folium/Streamlit.
    Returns URL template string {z}/{x}/{y}.
    """
    vis = get_index_vis_params(band_name)
    map_id = image.select(band_name).getMapId(vis)
    return map_id['tile_fetcher'].url_format
