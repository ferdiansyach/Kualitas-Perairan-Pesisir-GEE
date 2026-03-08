"""
06 - Machine Learning (Opsional)
================================
ML untuk prediksi kualitas air menggunakan sample data dari GEE.
Data di-sample dari server → training di laptop (data kecil).

Usage: python 06_machine_learning.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.gee_utils import (
    authenticate_gee, get_roi, get_sentinel2_collection,
    get_landsat8_sst_collection, sample_training_data, 
    ANALYSIS_YEARS, INDEX_NAMES
)
from utils.water_indices import ee_add_all_indices

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
CHARTS_DIR = os.path.join(BASE_DIR, 'output', 'charts')

# Fitur (band + rasio)
FEATURE_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11']
# Target (indeks kualitas air)
TARGET_PARAMS = ['NDCI', 'NDTI', 'TSS', 'CDOM', 'Secchi_Depth', 'SST']


def collect_training_data(n_points=5000):
    """Kumpulkan training data dari semua tahun via GEE sampling."""
    print("📋 Mengumpulkan training data dari GEE...\n")
    roi = get_roi()
    all_data = []

    for year in ANALYSIS_YEARS:
        print(f"   📅 Tahun {year}:")
        composite_s2 = get_sentinel2_collection(year, roi, max_cloud_pct=10)
        composite_l8 = get_landsat8_sst_collection(year, roi, max_cloud_pct=20)
        image = ee_add_all_indices(composite_s2).addBands(composite_l8)

        # Sample dari GEE — data kecil (beberapa MB)
        df = sample_training_data(image, n_points=n_points, scale=30, roi=roi)
        df['year'] = year
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.dropna()
    print(f"\n   ✅ Total: {len(combined)} samples berhasil dikumpulkan")

    # Simpan sebagai CSV
    csv_path = os.path.join(RESULTS_DIR, 'ml_training_data.csv')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    print(f"   💾 Data: {csv_path} ({os.path.getsize(csv_path)/1024:.0f} KB)")

    return combined


def train_models(df):
    """Latih model ML untuk setiap parameter target."""
    print(f"\n{'='*55}")
    print(f"🤖 MACHINE LEARNING")
    print(f"{'='*55}")

    # Filter fitur yang tersedia
    features = [b for b in FEATURE_BANDS if b in df.columns]
    # Tambah rasio spektral sebagai fitur tambahan
    if 'B2' in df.columns and 'B3' in df.columns:
        df['B2_B3_ratio'] = df['B2'] / df['B3'].replace(0, np.nan)
        features.append('B2_B3_ratio')
    if 'B4' in df.columns and 'B3' in df.columns:
        df['B4_B3_ratio'] = df['B4'] / df['B3'].replace(0, np.nan)
        features.append('B4_B3_ratio')
    if 'B5' in df.columns and 'B4' in df.columns:
        df['B5_B4_ratio'] = df['B5'] / df['B4'].replace(0, np.nan)
        features.append('B5_B4_ratio')

    df = df.dropna(subset=features)

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0)
    }

    all_results = {}

    for target in TARGET_PARAMS:
        if target not in df.columns:
            continue

        print(f"\n   📌 Target: {target}")
        data = df[features + [target]].dropna()

        if len(data) < 50:
            print(f"      ⚠️  Tidak cukup data ({len(data)} samples)")
            continue

        X = data[features].values
        y = data[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        target_results = {}

        for name, model in models.items():
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            target_results[name] = {
                'R2': round(r2, 4),
                'RMSE': round(rmse, 6),
                'MAE': round(mae, 6)
            }
            print(f"      {name:>20}: R²={r2:.4f}, RMSE={rmse:.6f}")

        # Feature importance (Random Forest)
        rf = models['Random Forest']
        if hasattr(rf, 'feature_importances_'):
            imp = dict(zip(features, [round(v, 4) for v in rf.feature_importances_]))
            target_results['feature_importance'] = imp

        all_results[target] = target_results

    return all_results, features


def plot_ml_results(results):
    """Visualisasi hasil ML."""
    os.makedirs(CHARTS_DIR, exist_ok=True)

    params = [p for p in TARGET_PARAMS if p in results]
    model_names = ['Random Forest', 'Gradient Boosting', 'SVR']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(params))
    width = 0.25

    for i, model in enumerate(model_names):
        r2_values = [results[p].get(model, {}).get('R2', 0) for p in params]
        ax.bar(x + i * width, r2_values, width, label=model, alpha=0.85)

    ax.set_xlabel('Parameter')
    ax.set_ylabel('R² Score')
    ax.set_title('Perbandingan Model ML — R² Score', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(params, rotation=15)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    path = os.path.join(CHARTS_DIR, 'ml_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\n   📊 R² comparison: {path}")
    plt.close()


def main():
    print("\n🤖 MACHINE LEARNING — Cloud Sample Processing\n")

    authenticate_gee()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Cek data yang sudah ada
    csv_path = os.path.join(RESULTS_DIR, 'ml_training_data.csv')
    if os.path.exists(csv_path):
        print(f"📂 Data training sudah ada: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = collect_training_data(n_points=5000)

    print(f"📋 Dataset: {len(df)} samples, {len(df.columns)} kolom\n")

    # Train models
    results, features = train_models(df)

    # Save results
    results_path = os.path.join(RESULTS_DIR, 'ml_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Evaluasi ML: {results_path}")

    # Plot
    plot_ml_results(results)

    print(f"\n✅ MACHINE LEARNING SELESAI!")
    print(f"   📊 Hasil: {RESULTS_DIR}/ml_evaluation.json")


if __name__ == '__main__':
    main()
