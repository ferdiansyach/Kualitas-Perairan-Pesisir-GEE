"""
03 - Cloud Analysis (Spasial-Temporal)
======================================
Analisis spasial-temporal sepenuhnya di GEE server:
- Change detection antar tahun
- Korelasi antar parameter
- Trend temporal
- Hotspot identification

Usage: python 03_cloud_analysis.py
"""

import ee
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.gee_utils import (
    authenticate_gee, get_roi, get_sentinel2_collection,
    get_landsat8_sst_collection, compute_region_stats, 
    save_thumbnail, get_index_vis_params,
    ANALYSIS_YEARS, INDEX_NAMES
)
from utils.water_indices import ee_add_all_indices

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
MAPS_DIR = os.path.join(BASE_DIR, 'output', 'maps')


def get_processed_image(year, roi):
    """Ambil composite + indices + sst dari GEE."""
    composite_s2 = get_sentinel2_collection(year, roi, max_cloud_pct=10)
    composite_l8 = get_landsat8_sst_collection(year, roi, max_cloud_pct=20)
    return ee_add_all_indices(composite_s2).addBands(composite_l8)


def change_detection(img_early, img_late, year_early, year_late, roi):
    """
    Deteksi perubahan antara dua tahun.
    Dilakukan di GEE server — hanya return dict kecil.
    """
    print(f"\n🔄 Change Detection: {year_early} → {year_late}")
    results = {}

    for param in INDEX_NAMES:
        try:
            early = img_early.select(param)
            late = img_late.select(param)
            diff = late.subtract(early).rename('change')

            # Statistik perubahan
            stats = diff.reduceRegion(
                reducer=ee.Reducer.mean()
                    .combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=roi, scale=30,
                maxPixels=1e9, bestEffort=True
            ).getInfo()

            # Persentase area meningkat vs menurun
            increased = diff.gt(0).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi,
                scale=30, maxPixels=1e9, bestEffort=True
            ).getInfo()

            key = list(increased.keys())[0] if increased else 'change'
            pct_inc = (increased.get(key, 0) or 0) * 100

            results[param] = {
                'mean_change': stats.get('change_mean', 0) or 0,
                'std_change': stats.get('change_stdDev', 0) or 0,
                'pct_increased': round(pct_inc, 1),
                'pct_decreased': round(100 - pct_inc, 1),
            }
            print(f"   {param}: Δmean = {results[param]['mean_change']:+.4f}, "
                  f"↑{results[param]['pct_increased']:.0f}%")

            # Simpan peta perubahan sebagai thumbnail
            vis_params = {
                'min': -0.2, 'max': 0.2,
                'palette': ['red', 'white', 'green'],
                'region': roi, 'dimensions': 800, 'format': 'png'
            }
            try:
                url = diff.getThumbURL(vis_params)
                import requests
                resp = requests.get(url, timeout=60)
                path = os.path.join(MAPS_DIR, f'change_{param}_{year_early}_{year_late}.png')
                os.makedirs(MAPS_DIR, exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(resp.content)
            except:
                pass

        except Exception as e:
            print(f"   ⚠️  {param}: {e}")
            results[param] = {'mean_change': 0, 'std_change': 0,
                              'pct_increased': 50, 'pct_decreased': 50}

    return results


def correlation_analysis(image, roi):
    """
    Hitung korelasi antar parameter di GEE.
    Menggunakan sample points.
    """
    print(f"\n📐 Analisis Korelasi antar parameter...")

    # Sample points
    samples = image.select(INDEX_NAMES).sample(
        region=roi, scale=30, numPixels=5000,
        seed=42, geometries=False
    )

    data = samples.getInfo()
    features = data.get('features', [])
    if not features:
        print("   ⚠️  Tidak cukup sample untuk korelasi")
        return {}

    # Extract ke numpy
    import pandas as pd
    rows = [f['properties'] for f in features]
    df = pd.DataFrame(rows).dropna()

    if df.empty:
        return {}

    # Correlation matrix
    corr = df[INDEX_NAMES].corr().round(4).to_dict()
    print(f"   ✅ Korelasi dihitung dari {len(df)} samples")

    # Print summary
    for p1 in INDEX_NAMES[:3]:
        for p2 in INDEX_NAMES[:3]:
            if p1 < p2:
                val = corr.get(p1, {}).get(p2, 0)
                print(f"      {p1} ↔ {p2}: {val:.3f}")

    return corr


def temporal_trend(all_stats, output_dir):
    """Analisis trend temporal dari statistik yang sudah dihitung."""
    print(f"\n📈 Analisis Trend Temporal...")

    import pandas as pd
    rows = []
    for year in ANALYSIS_YEARS:
        stat_file = os.path.join(RESULTS_DIR, f'statistics_{year}.json')
        if os.path.exists(stat_file):
            with open(stat_file) as f:
                data = json.load(f)
            for param, vals in data['statistics'].items():
                rows.append({
                    'year': year, 'parameter': param,
                    'mean': vals.get('mean', 0),
                    'std': vals.get('stdDev', 0),
                    'min': vals.get('min', 0),
                    'max': vals.get('max', 0),
                })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, 'temporal_trends.csv')
        df.to_csv(csv_path, index=False)
        print(f"   💾 Trend data: {csv_path}")

        from scipy.stats import linregress
        trend_stats = []

        # Simple linear trend & P-Value
        for param in INDEX_NAMES:
            sub = df[df['parameter'] == param].sort_values('year')
            if len(sub) >= 3:
                first = sub.iloc[0]['mean']
                last = sub.iloc[-1]['mean']
                change = last - first
                direction = "↑" if change > 0 else "↓"
                
                # Uji Signifikansi Statistik (P-value via Linear Regression)
                slope, intercept, r_val, p_val, stderr = linregress(sub['year'], sub['mean'])
                sig_label = "✅ Signifikan" if p_val < 0.05 else "❌ Tidak Signifikan"

                print(f"   {param}: {first:.4f} → {last:.4f} ({direction}{abs(change):.4f}) | P-val: {p_val:.3f} ({sig_label})")
                
                trend_stats.append({
                    'parameter': param,
                    'slope_per_year': slope,
                    'p_value': p_val,
                    'significance': "Significant" if p_val < 0.05 else "Not Significant"
                })

        # Save trend significance
        if trend_stats:
            pd.DataFrame(trend_stats).to_csv(os.path.join(output_dir, 'trend_significance.csv'), index=False)

    return rows


def main():
    print("\n" + "📊 " * 20)
    print("  CLOUD ANALYSIS — Spasial-Temporal")
    print("  Pesisir Jakarta & Banten")
    print("📊 " * 20)

    authenticate_gee()
    roi = get_roi()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)

    # Load images dari GEE
    print(f"\n☁️  Memuat data dari GEE server...")
    images = {}
    for year in ANALYSIS_YEARS:
        images[year] = get_processed_image(year, roi)

    # 1. Change Detection
    year_first = ANALYSIS_YEARS[0]
    year_last = ANALYSIS_YEARS[-1]
    change_results = change_detection(
        images[year_first], images[year_last],
        year_first, year_last, roi
    )

    change_path = os.path.join(RESULTS_DIR, 'change_detection.json')
    with open(change_path, 'w') as f:
        json.dump({
            'period': f'{year_first}-{year_last}',
            'parameters': change_results
        }, f, indent=2, default=str)
    print(f"   💾 Change detection: {change_path}")

    # 2. Korelasi (gunakan tahun terakhir)
    corr = correlation_analysis(images[year_last], roi)
    corr_path = os.path.join(RESULTS_DIR, 'correlation_matrix.json')
    with open(corr_path, 'w') as f:
        json.dump(corr, f, indent=2, default=str)

    # 3. Trend Temporal
    temporal_trend({}, RESULTS_DIR)

    print(f"\n✅ CLOUD ANALYSIS SELESAI!")
    print(f"   📊 Hasil: {RESULTS_DIR}/")
    print(f"   🗺️  Peta: {MAPS_DIR}/")
    print(f"\n   → Jalankan: python 04_visualization.py")


if __name__ == '__main__':
    main()
