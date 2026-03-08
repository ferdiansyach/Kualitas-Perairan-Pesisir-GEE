"""
04 - Visualization
==================
Membuat grafik dan visualisasi dari hasil analisis (JSON/CSV).
Tidak memerlukan GEE — bekerja dari file lokal kecil.

Usage: python 04_visualization.py
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.gee_utils import ANALYSIS_YEARS, INDEX_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
MAPS_DIR = os.path.join(BASE_DIR, 'output', 'maps')
CHARTS_DIR = os.path.join(BASE_DIR, 'output', 'charts')

PARAM_LABELS = {
    'NDCI': 'Chlorophyll-a (NDCI)',
    'NDTI': 'Turbiditas (NDTI)',
    'TSS': 'TSS (mg/L)',
    'CDOM': 'CDOM (B2/B3)',
    'Secchi_Depth': 'Secchi Depth (m)',
    'SST': 'Suhu Perairan (SST °C)',
}

COLORS = ['#0288D1', '#388E3C', '#F57C00', '#D32F2F', '#7B1FA2', '#E64A19']
plt.rcParams['font.family'] = 'sans-serif'


def load_all_stats():
    """Load semua file statistik JSON."""
    rows = []
    for year in ANALYSIS_YEARS:
        path = os.path.join(RESULTS_DIR, f'statistics_{year}.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for param, vals in data['statistics'].items():
                rows.append({
                    'Year': year, 'Parameter': param,
                    'Mean': vals.get('mean', 0) or 0,
                    'Std': vals.get('stdDev', 0) or 0,
                    'Min': vals.get('min', 0) or 0,
                    'Max': vals.get('max', 0) or 0,
                    'Coverage': vals.get('coverage_pct', 0) or 0,
                })
    return pd.DataFrame(rows)


def plot_temporal_trends(df):
    """Grafik trend temporal per parameter."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, param in enumerate(INDEX_NAMES):
        if i >= len(axes):
            break
        ax = axes[i]
        sub = df[df['Parameter'] == param].sort_values('Year')

        ax.plot(sub['Year'], sub['Mean'], 'o-', color=COLORS[i],
                linewidth=2.5, markersize=8, label='Mean')
        ax.fill_between(sub['Year'],
                        sub['Mean'] - sub['Std'],
                        sub['Mean'] + sub['Std'],
                        alpha=0.2, color=COLORS[i])

        ax.set_title(PARAM_LABELS.get(param, param),
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Nilai')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ANALYSIS_YEARS)

    for j in range(len(INDEX_NAMES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Trend Temporal Parameter Kualitas Air\n'
                 'Pesisir Jakarta & Banten',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, 'temporal_trends.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"   📈 {path}")
    plt.close()


def plot_bar_comparison(df):
    """Bar chart perbandingan antar tahun."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, param in enumerate(INDEX_NAMES):
        if i >= len(axes):
            break
        ax = axes[i]
        sub = df[df['Parameter'] == param].sort_values('Year')

        bars = ax.bar([str(y) for y in sub['Year']], sub['Mean'],
                      color=COLORS[i], alpha=0.85, edgecolor='white',
                      linewidth=1.5)
        ax.errorbar([str(y) for y in sub['Year']], sub['Mean'],
                    yerr=sub['Std'], fmt='none', capsize=5,
                    color='black', alpha=0.5)

        ax.set_title(PARAM_LABELS.get(param, param),
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Tahun')
        ax.grid(True, axis='y', alpha=0.3)

    for j in range(len(INDEX_NAMES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Perbandingan Parameter Kualitas Air per Tahun',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, 'bar_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"   📊 {path}")
    plt.close()


def plot_correlation_heatmap():
    """Heatmap korelasi antar parameter."""
    corr_path = os.path.join(RESULTS_DIR, 'correlation_matrix.json')
    if not os.path.exists(corr_path):
        print("   ⚠️  Tidak ada data korelasi")
        return

    with open(corr_path) as f:
        corr_dict = json.load(f)

    if not corr_dict:
        return

    df = pd.DataFrame(corr_dict)
    # Reorder
    cols = [c for c in INDEX_NAMES if c in df.columns]
    df = df.reindex(index=cols, columns=cols)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=1, ax=ax,
                fmt='.3f', vmin=-1, vmax=1)
    ax.set_title('Korelasi Antar Parameter Kualitas Air',
                 fontsize=13, fontweight='bold', pad=15)

    path = os.path.join(CHARTS_DIR, 'correlation_heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"   🔥 {path}")
    plt.close()


def plot_change_detection():
    """Visualisasi hasil change detection."""
    cd_path = os.path.join(RESULTS_DIR, 'change_detection.json')
    if not os.path.exists(cd_path):
        return

    with open(cd_path) as f:
        data = json.load(f)

    params = data.get('parameters', {})
    if not params:
        return

    names = list(params.keys())
    changes = [params[n]['mean_change'] for n in names]
    pct_inc = [params[n]['pct_increased'] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar: mean change
    colors_bar = ['green' if c > 0 else 'red' for c in changes]
    ax1.barh(names, changes, color=colors_bar, alpha=0.8, edgecolor='white')
    ax1.set_xlabel('Perubahan Mean')
    ax1.set_title(f"Perubahan {data['period']}", fontweight='bold')
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.grid(True, axis='x', alpha=0.3)

    # Bar: % increase
    ax2.barh(names, pct_inc, color='#0288D1', alpha=0.8)
    ax2.set_xlabel('% Area Meningkat')
    ax2.set_title('Persentase Area Meningkat', fontweight='bold')
    ax2.axvline(50, color='gray', linewidth=0.8, linestyle='--')
    ax2.set_xlim(0, 100)
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, 'change_detection.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"   🔄 {path}")
    plt.close()


def create_map_mosaic():
    """Buat mosaic dari thumbnail peta yang sudah di-download."""
    from PIL import Image

    for year in ANALYSIS_YEARS:
        imgs = []
        labels = []
        for param in INDEX_NAMES:
            path = os.path.join(MAPS_DIR, f'{param}_{year}.png')
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    imgs.append(Image.open(path))
                    labels.append(param)
                except Exception as e:
                    print(f"   ⚠️  Skipping corrupted image {path}: {e}")

        if not imgs:
            continue

        # Grid layout
        n = len(imgs)
        cols = min(3, n)
        rows_n = (n + cols - 1) // cols
        w, h = imgs[0].size

        mosaic = Image.new('RGB', (w * cols, h * rows_n), (255, 255, 255))
        for idx, img in enumerate(imgs):
            r, c = divmod(idx, cols)
            mosaic.paste(img.resize((w, h)), (c * w, r * h))

        path = os.path.join(MAPS_DIR, f'mosaic_{year}.png')
        mosaic.save(path)
        print(f"   🖼️  Mosaic {year}: {path}")


def main():
    print("\n🎨 VISUALISASI — Perairan Pesisir Jakarta & Banten\n")

    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)

    # Load data
    df = load_all_stats()
    if df.empty:
        print("❌ Tidak ada data statistik. Jalankan 02_cloud_processing.py dulu.")
        return

    print(f"📋 Data: {len(df)} records dari {df['Year'].nunique()} tahun\n")

    # 1. Trend temporal
    print("📈 Membuat grafik trend temporal...")
    plot_temporal_trends(df)

    # 2. Bar comparison
    print("📊 Membuat perbandingan bar chart...")
    plot_bar_comparison(df)

    # 3. Correlation heatmap
    print("🔥 Membuat heatmap korelasi...")
    plot_correlation_heatmap()

    # 4. Change detection chart
    print("🔄 Membuat grafik change detection...")
    plot_change_detection()

    # 5. Map mosaic
    print("🖼️  Membuat mosaic peta...")
    try:
        create_map_mosaic()
    except ImportError:
        print("   ⚠️  Pillow belum terinstall — skip mosaic")

    print(f"\n✅ VISUALISASI SELESAI!")
    print(f"   📊 Charts: {CHARTS_DIR}/")
    print(f"   🗺️  Peta: {MAPS_DIR}/")
    print(f"\n   → Jalankan: streamlit run 05_dashboard.py")


if __name__ == '__main__':
    main()
