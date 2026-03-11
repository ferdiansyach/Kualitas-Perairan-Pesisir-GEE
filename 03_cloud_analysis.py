"""
03 - Cloud Analysis (Spasial-Temporal)
======================================
Analisis spasial-temporal sepenuhnya di GEE server:
- Change detection antar tahun
- Korelasi antar parameter
- Trend temporal (Mann-Kendall + Sen's Slope)
- Hotspot identification

CATATAN METODOLOGI:
  Versi sebelumnya menggunakan scipy.stats.linregress untuk uji tren.
  Dengan N=7 titik data, asumsi normalitas residual tidak dapat
  diverifikasi, sehingga p-value parametrik tidak valid.

  Solusi: Mann-Kendall Trend Test + Sen's Slope Estimator.
  Keduanya adalah uji non-parametrik, standar dalam hidrologi dan
  analisis deret-waktu klimatik (Hamed & Rao, 1998).

  Instalasi dependensi: pip install pymannkendall

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


def _mann_kendall_fallback(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Implementasi Mann-Kendall manual sebagai fallback jika pymannkendall
    tidak terinstal. Menghitung statistik S dan Sen's Slope.
    Referensi: Mann (1945), Kendall (1975).
    """
    n = len(y)
    # Statistik S
    S = sum(
        np.sign(y[j] - y[i])
        for i in range(n - 1)
        for j in range(i + 1, n)
    )
    # Variansi S (tanpa koreksi ties)
    var_S = n * (n - 1) * (2 * n + 5) / 18
    # Z-score
    if S > 0:
        z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_S)
    else:
        z = 0.0
    # p-value (dua sisi)
    from scipy.special import erfc
    p_value = erfc(abs(z) / np.sqrt(2))
    # Sen's Slope
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(n - 1)
        for j in range(i + 1, n)
        if x[j] != x[i]
    ]
    sen_slope = float(np.median(slopes)) if slopes else 0.0

    trend = 'increasing' if S > 0 else ('decreasing' if S < 0 else 'no trend')
    return {
        'trend': trend,
        'h': p_value < 0.05,
        'p': float(p_value),
        'z': float(z),
        'Tau': float(S / (n * (n - 1) / 2)),
        'slope': sen_slope,
        'method': 'Mann-Kendall (manual)',
    }


def mann_kendall_trend(series: np.ndarray, years: list) -> dict:
    """
    Uji tren Mann-Kendall + Sen's Slope Estimator.

    Menggunakan pymannkendall jika tersedia (direkomendasikan), atau
    implementasi manual sebagai fallback.

    Mengapa Mann-Kendall lebih baik daripada linregress di sini:
    - Non-parametrik: tidak mengasumsikan distribusi normal
    - Robust terhadap outlier (misalnya tahun ekstrem akibat El Niño)
    - Standar dalam analisis tren hidrologis dan klimatik
    - Valid untuk N kecil (N ≥ 4 sudah dapat diinterpretasikan)
    """
    x = np.array(years, dtype=float)
    y = np.array(series, dtype=float)

    try:
        import pymannkendall as mk
        result = mk.original_test(y)
        # Sen's Slope (estimator lereng median-based — robust terhadap outlier)
        slopes = [
            (y[j] - y[i]) / (x[j] - x[i])
            for i in range(len(y) - 1)
            for j in range(i + 1, len(y))
            if x[j] != x[i]
        ]
        sen_slope = float(np.median(slopes)) if slopes else 0.0
        return {
            'trend': result.trend,
            'h': result.h,          # True jika signifikan pada α=0.05
            'p': float(result.p),
            'z': float(result.z),
            'Tau': float(result.Tau),
            'slope': sen_slope,
            'method': 'Mann-Kendall (pymannkendall)',
        }
    except ImportError:
        # Fallback ke implementasi manual jika library tidak tersedia
        return _mann_kendall_fallback(x, y)


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
            # FIX: Kembalikan None/NaN — bukan 0.
            # Nilai 0 pada parameter fisik (mis. SST=0°C, TSS=0) akan
            # mendistorsi rata-rata tahunan dan seluruh analisis tren.
            results[param] = {
                'mean_change': None,
                'std_change': None,
                'pct_increased': None,
                'pct_decreased': None,
            }

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
    """
    Analisis trend temporal menggunakan Mann-Kendall + Sen's Slope.

    Mengapa bukan linregress:
    - N=7 terlalu kecil untuk memvalidasi asumsi normalitas
    - Satu tahun ekstrem (El Niño, banjir besar) dapat mengubah slope
      secara drastis pada regresi parametrik
    - Mann-Kendall adalah standar WMO untuk tren hidrologis jangka pendek
    """
    print(f"\n📈 Analisis Trend Temporal (Mann-Kendall + Sen's Slope)...")

    import pandas as pd

    rows = []
    for year in ANALYSIS_YEARS:
        stat_file = os.path.join(RESULTS_DIR, f'statistics_{year}.json')
        if os.path.exists(stat_file):
            with open(stat_file) as f:
                data = json.load(f)
            for param, vals in data['statistics'].items():
                # FIX: Gunakan None/NaN untuk nilai yang gagal dihitung,
                # bukan 0. Ini mencegah distorsi pada perhitungan tren.
                mean_val = vals.get('mean')
                rows.append({
                    'year': year, 'parameter': param,
                    'mean': mean_val,         # None jika gagal
                    'std': vals.get('stdDev'),
                    'min': vals.get('min'),
                    'max': vals.get('max'),
                })

    if not rows:
        print("   ⚠️  Tidak ada data statistik. Jalankan 02_cloud_processing.py dulu.")
        return []

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'temporal_trends.csv')
    df.to_csv(csv_path, index=False)
    print(f"   💾 Trend data: {csv_path}")

    trend_results = []
    print(f"\n   {'Parameter':<15} {'Trend':>12} {'Sen Slope':>12} {'p-value':>10} {'Signifikan':>12}")
    print(f"   {'-'*65}")

    for param in INDEX_NAMES:
        sub = df[(df['parameter'] == param) & (df['mean'].notna())].sort_values('year')

        if len(sub) < 4:
            # Mann-Kendall memerlukan minimal 4 titik untuk bermakna
            print(f"   {param:<15} {'N/A (data kurang)':>40}")
            continue

        mk_result = mann_kendall_trend(sub['mean'].values, sub['year'].tolist())

        sig_label = "✅ Ya" if mk_result['h'] else "❌ Tidak"
        print(f"   {param:<15} {mk_result['trend']:>12} "
              f"{mk_result['slope']:>+12.6f} "
              f"{mk_result['p']:>10.4f} "
              f"{sig_label:>12}")

        trend_results.append({
            'parameter': param,
            'trend_direction': mk_result['trend'],
            'sens_slope_per_year': mk_result['slope'],
            'kendall_tau': mk_result['Tau'],
            'z_score': mk_result['z'],
            'p_value': mk_result['p'],
            'significant_alpha005': mk_result['h'],
            'method': mk_result['method'],
            'n_years': len(sub),
        })

    if trend_results:
        trend_df = pd.DataFrame(trend_results)
        trend_path = os.path.join(output_dir, 'trend_significance.csv')
        trend_df.to_csv(trend_path, index=False)
        print(f"\n   💾 Signifikansi tren: {trend_path}")

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
