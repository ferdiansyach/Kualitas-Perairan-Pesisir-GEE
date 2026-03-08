"""
Water Quality Indices (Cloud + Local)
=====================================
Fungsi-fungsi untuk menghitung parameter kualitas perairan
dari citra satelit Sentinel-2, baik di GEE maupun lokal.

Referensi:
- NDCI: Mishra & Mishra (2012) - Chlorophyll-a estimation
- NDTI: Lacaux et al. (2007) - Turbidity estimation
- NDWI: McFeeters (1996) - Water body detection
- TSS: Petus et al. (2010) - Suspended solids
"""

import ee


# ============================================================
# EARTH ENGINE — Komputasi di Cloud
# ============================================================

def ee_calculate_ndwi(image):
    """NDWI: (Green - NIR) / (Green + NIR) → deteksi air."""
    return image.normalizedDifference(['B3', 'B8']).toFloat().rename('NDWI')


def ee_calculate_mndwi(image):
    """MNDWI: (Green - SWIR) / (Green + SWIR)."""
    return image.normalizedDifference(['B3', 'B11']).toFloat().rename('MNDWI')


def ee_calculate_ndci(image):
    """NDCI: (RedEdge1 - Red) / (RedEdge1 + Red) → Chlorophyll-a."""
    return image.normalizedDifference(['B5', 'B4']).toFloat().rename('NDCI')


def ee_calculate_ndti(image):
    """NDTI: (Red - Green) / (Red + Green) → Turbidity."""
    return image.normalizedDifference(['B4', 'B3']).toFloat().rename('NDTI')


def ee_calculate_tss(image):
    """
    TSS (Total Suspended Solids) estimation.
    TSS = 3983.9 * Red_reflectance^1.6246  (Petus et al. 2010)
    """
    red = image.select('B4').divide(10000)  # DN to reflectance
    tss = red.pow(1.6246).multiply(3983.9).toFloat().rename('TSS')
    return tss


def ee_calculate_cdom(image):
    """
    CDOM (Colored Dissolved Organic Matter).
    Rasio Blue/Green sebagai proxy. Nilai rendah = CDOM tinggi.
    """
    blue = image.select('B2').toFloat()
    green = image.select('B3').toFloat()
    cdom = blue.divide(green).rename('CDOM')
    return cdom


def ee_calculate_secchi(image):
    """
    Secchi Disk Depth estimation.
    Formula: ln(Blue) / ln(Green) (simplified Lee et al. model)
    """
    blue = image.select('B2').toFloat().divide(10000).max(0.001)
    green = image.select('B3').toFloat().divide(10000).max(0.001)
    secchi = blue.log().divide(green.log()).rename('Secchi_Depth')
    return secchi


def ee_create_water_mask(image, threshold=0.0):
    """Buat water mask dari NDWI. True = air."""
    ndwi = ee_calculate_ndwi(image)
    return ndwi.gt(threshold).rename('water_mask')


def ee_add_all_indices(image):
    """
    Tambahkan SEMUA indeks kualitas air ke image sebagai band baru.
    Hanya area air (NDWI > 0) yang memiliki nilai.
    Returns ee.Image dengan band asli + 7 indeks baru.
    """
    # Hitung indices
    ndwi = ee_calculate_ndwi(image)
    mndwi = ee_calculate_mndwi(image)
    ndci = ee_calculate_ndci(image)
    ndti = ee_calculate_ndti(image)
    tss = ee_calculate_tss(image)
    cdom = ee_calculate_cdom(image)
    secchi = ee_calculate_secchi(image)

    # Water mask (diperketat > 0.1 untuk menghindari tanah basah/tambak dangkal)
    water_mask = ndwi.gt(0.1)

    # Masking — hanya air
    ndci_masked = ndci.updateMask(water_mask)
    ndti_masked = ndti.updateMask(water_mask)
    tss_masked = tss.updateMask(water_mask)
    cdom_masked = cdom.updateMask(water_mask)
    secchi_masked = secchi.updateMask(water_mask)

    return image.addBands([
        ndwi, mndwi,
        ndci_masked, ndti_masked,
        tss_masked, cdom_masked, secchi_masked
    ])


# ============================================================
# NUMPY — Komputasi Lokal (untuk data yang sudah di-download)
# ============================================================

import numpy as np


def safe_normalized_difference(a, b):
    """Hitung normalized difference secara aman."""
    denominator = a.astype(np.float64) + b.astype(np.float64)
    result = np.where(denominator != 0,
                      (a.astype(np.float64) - b.astype(np.float64)) / denominator,
                      np.nan)
    return result.astype(np.float32)


def calculate_ndwi(green, nir):
    """NDWI lokal: (Green - NIR) / (Green + NIR)"""
    return safe_normalized_difference(green, nir)


def calculate_ndci(red_edge1, red):
    """NDCI lokal: (RedEdge1 - Red) / (RedEdge1 + Red)"""
    return safe_normalized_difference(red_edge1, red)


def calculate_ndti(red, green):
    """NDTI lokal: (Red - Green) / (Red + Green)"""
    return safe_normalized_difference(red, green)


def calculate_tss(red):
    """TSS lokal: 3983.9 * Red^1.6246"""
    red_refl = np.where(red > 1, red / 10000.0, red).astype(np.float64)
    tss = np.where(red_refl > 0, 3983.9 * np.power(red_refl, 1.6246), np.nan)
    return tss.astype(np.float32)


def calculate_cdom(blue, green):
    """CDOM lokal: Blue / Green"""
    green_f = green.astype(np.float64)
    result = np.where(green_f > 0, blue.astype(np.float64) / green_f, np.nan)
    return result.astype(np.float32)


def calculate_secchi_depth(blue, green):
    """Secchi Depth lokal: ln(Blue) / ln(Green)"""
    blue_f = blue.astype(np.float64)
    green_f = green.astype(np.float64)
    valid = (blue_f > 0) & (green_f > 0)
    result = np.where(valid, np.log(blue_f) / np.log(green_f), np.nan)
    return result.astype(np.float32)
