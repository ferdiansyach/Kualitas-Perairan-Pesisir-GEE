"""
02 - Cloud Processing (GEE)
===========================
Semua analisis kualitas air berjalan di server Google Earth Engine.
Laptop hanya menerima hasil kecil: statistik (JSON) dan peta (PNG).

TIDAK ADA download file GeoTIFF besar.

Usage: python 02_cloud_processing.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.gee_utils import (
    authenticate_gee, get_roi, get_sentinel2_collection,
    get_landsat8_sst_collection, compute_all_stats, 
    save_stats_json, generate_all_thumbnails,
    ANALYSIS_YEARS, INDEX_NAMES
)
from utils.water_indices import ee_add_all_indices

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
MAPS_DIR = os.path.join(BASE_DIR, 'output', 'maps')


def process_year(year, roi):
    """Proses satu tahun: komposit → indeks → statistik → peta."""
    print(f"\n{'='*55}")
    print(f"📅  TAHUN {year}")
    print(f"{'='*55}")

    # 1. Komposit Sentinel-2 (di GEE)
    composite_s2 = get_sentinel2_collection(year, roi, max_cloud_pct=10)

    # 1b. Komposit Landsat-8 untuk SST (di GEE)
    composite_l8_sst = get_landsat8_sst_collection(year, roi, max_cloud_pct=20)

    # 2. Hitung semua indeks kualitas air (di GEE) & Gabungkan dengan SST
    print(f"   🔬 Menghitung indeks kualitas air...")
    image_with_indices = ee_add_all_indices(composite_s2).addBands(composite_l8_sst)

    # 3. Hitung statistik per parameter (di GEE, return dict kecil)
    print(f"   📊 Menghitung statistik...")
    stats = compute_all_stats(image_with_indices, INDEX_NAMES, roi, scale=30)

    # 4. Simpan statistik sebagai JSON (beberapa KB)
    save_stats_json(stats, year, RESULTS_DIR)

    # 5. Generate thumbnail peta (PNG ~100KB per peta)
    print(f"   🗺️  Generating peta thumbnail...")
    generate_all_thumbnails(image_with_indices, year, MAPS_DIR, dimensions=800)

    return image_with_indices, stats


def generate_summary(all_stats):
    """Buat ringkasan gabungan semua tahun."""
    summary_path = os.path.join(RESULTS_DIR, 'summary_all_years.json')
    with open(summary_path, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\n💾 Ringkasan: {summary_path}")


def main():
    print("\n" + "🌊 " * 20)
    print("  CLOUD PROCESSING — Kualitas Perairan Pesisir")
    print("  Jakarta & Banten (Full GEE)")
    print("🌊 " * 20)

    # Setup
    authenticate_gee()
    roi = get_roi()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)

    print(f"\n📋 Tahun analisis: {ANALYSIS_YEARS}")
    print(f"📊 Parameter: {', '.join(INDEX_NAMES)}")
    print(f"☁️  Semua komputasi di server GEE — laptop Anda tetap ringan!\n")

    # Proses setiap tahun
    all_stats = {}
    all_images = {}

    for year in ANALYSIS_YEARS:
        image, stats = process_year(year, roi)
        all_stats[year] = stats
        all_images[year] = image

    # Ringkasan
    generate_summary(all_stats)

    # Summary table
    print(f"\n{'='*60}")
    print(f"📊 RINGKASAN STATISTIK")
    print(f"{'='*60}")
    print(f"{'Parameter':<15}", end="")
    for y in ANALYSIS_YEARS:
        print(f"  {y:>10}", end="")
    print()
    print("-" * 60)

    for param in INDEX_NAMES:
        print(f"{param:<15}", end="")
        for year in ANALYSIS_YEARS:
            val = all_stats.get(year, {}).get(param, {}).get('mean', 0)
            print(f"  {val:>10.4f}", end="")
        print()

    print(f"\n✅ CLOUD PROCESSING SELESAI!")
    print(f"   📊 Statistik: {RESULTS_DIR}/")
    print(f"   🗺️  Peta: {MAPS_DIR}/")
    print(f"\n   → Jalankan: python 03_cloud_analysis.py")


if __name__ == '__main__':
    main()
