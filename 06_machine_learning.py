"""
06 - Machine Learning (Unsupervised Clustering)
================================================
Menggunakan K-Means untuk mengklasifikasikan zona kualitas air
berdasarkan multi-band signature satelit.

CATATAN METODOLOGI:
  Versi sebelumnya melatih model untuk memprediksi NDCI/NDTI/TSS dari
  band spektral — ini adalah circular logic karena indeks tersebut
  dihitung deterministik dari band yang sama (R²≈0.99 tidak bermakna).

  Solusi: Gunakan Unsupervised Clustering (K-Means) untuk
  mengelompokkan piksel air ke zona "Bersih", "Sedang", "Tercemar"
  berdasarkan signature spektral multi-band. Ini valid secara ilmiah
  karena tidak ada ground-truth yang di-buat-ulang dari data yang sama.

  Jika Anda memiliki data in-situ (pengukuran lapangan), ganti variabel
  `y` dengan nilai ground-truth tersebut untuk supervised learning.

Usage: python 06_machine_learning.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

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

# Band spektral sebagai fitur (bebas dari circular logic)
FEATURE_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11']

# Label zona kualitas air (dari nilai indeks rendah → tinggi)
CLUSTER_LABELS = {
    0: {'name': 'Bersih',    'color': '#0288D1'},
    1: {'name': 'Sedang',    'color': '#FFA726'},
    2: {'name': 'Tercemar',  'color': '#D32F2F'},
}

N_CLUSTERS = 3  # Dapat disesuaikan berdasarkan analisis silhouette


def collect_spectral_samples(n_points: int = 5000) -> pd.DataFrame:
    """
    Sample pixel spektral dari GEE untuk semua tahun analisis.
    Hanya mengambil band reflektansi — BUKAN indeks turunan.
    """
    print("📡 Sampling data spektral dari GEE...\n")
    roi = get_roi()
    all_data = []

    for year in ANALYSIS_YEARS:
        print(f"   📅 Tahun {year}:")
        composite_s2 = get_sentinel2_collection(year, roi, max_cloud_pct=10)
        composite_l8 = get_landsat8_sst_collection(year, roi, max_cloud_pct=20)

        # Ambil band mentah — indeks TIDAK digunakan sebagai fitur
        image = composite_s2.addBands(composite_l8)
        df = sample_training_data(image, n_points=n_points, scale=30, roi=roi)
        df['year'] = year
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.dropna(subset=FEATURE_BANDS)

    print(f"\n   ✅ Total: {len(combined)} piksel berhasil di-sample")
    csv_path = os.path.join(RESULTS_DIR, 'ml_spectral_samples.csv')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    print(f"   💾 Data: {csv_path}")
    return combined


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 8)) -> int:
    """
    Tentukan K optimal menggunakan Silhouette Score dan Davies-Bouldin Index.
    Kedua metrik ini tidak memerlukan ground-truth.
    """
    print("\n🔍 Menentukan jumlah cluster optimal (K)...")
    silhouette_scores = []
    db_scores = []

    # Sub-sample untuk efisiensi
    n_eval = min(10_000, len(X_scaled))
    idx = np.random.choice(len(X_scaled), n_eval, replace=False)
    X_eval = X_scaled[idx]

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_eval)
        sil = silhouette_score(X_eval, labels, sample_size=min(5000, n_eval))
        db = davies_bouldin_score(X_eval, labels)
        silhouette_scores.append(sil)
        db_scores.append(db)
        print(f"   K={k}: Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}")

    # Pilih K dengan Silhouette tertinggi
    best_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"\n   ✅ K optimal = {best_k} (Silhouette = {max(silhouette_scores):.4f})")
    return best_k, silhouette_scores, list(k_range)


def run_kmeans_clustering(df: pd.DataFrame, n_clusters: int = N_CLUSTERS):
    """
    Latih K-Means pada data spektral dan simpan label cluster.
    """
    print(f"\n{'='*55}")
    print(f"🤖 K-MEANS CLUSTERING (K={n_clusters})")
    print(f"{'='*55}")

    features = [b for b in FEATURE_BANDS if b in df.columns]
    X = df[features].values

    # Pipeline: RobustScaler → K-Means
    # RobustScaler dipilih karena data piksel satelit sering mengandung outlier
    # akibat sisa awan tipis atau spike polusi lokal.
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=20)),
    ])
    pipeline.fit(X)
    labels = pipeline.predict(X)

    df = df.copy()
    df['cluster'] = labels

    # Beri label semantik berdasarkan rata-rata B4 (proxy turbiditas/TSS)
    # Cluster dengan B4 tertinggi → "Tercemar", terendah → "Bersih"
    cluster_b4_mean = df.groupby('cluster')['B4'].mean().sort_values()
    rank_map = {orig: new for new, orig in enumerate(cluster_b4_mean.index)}
    df['cluster_ranked'] = df['cluster'].map(rank_map)
    df['cluster_label'] = df['cluster_ranked'].map(
        lambda x: CLUSTER_LABELS.get(x, {}).get('name', f'Cluster {x}'))

    # Statistik per cluster
    print("\n   📊 Statistik per Cluster:")
    cluster_stats = {}
    for c in sorted(df['cluster_ranked'].unique()):
        sub = df[df['cluster_ranked'] == c]
        label = CLUSTER_LABELS.get(c, {}).get('name', f'Cluster {c}')
        pct = len(sub) / len(df) * 100
        b4_mean = sub['B4'].mean()
        print(f"   [{c}] {label:12s}: {len(sub):>6} piksel ({pct:4.1f}%) | B4 mean={b4_mean:.4f}")
        cluster_stats[label] = {
            'n_pixels': int(len(sub)),
            'pct': round(pct, 2),
            'mean_B4': round(float(b4_mean), 6),
            'mean_B3': round(float(sub['B3'].mean()), 6),
        }

    return df, pipeline, cluster_stats


def analyze_temporal_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung distribusi cluster per tahun untuk analisis temporal."""
    print("\n📅 Distribusi Zona per Tahun:")
    dist = (df.groupby(['year', 'cluster_label'])
              .size()
              .unstack(fill_value=0))
    dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100

    print(dist_pct.round(1).to_string())
    return dist_pct


def plot_results(df: pd.DataFrame, silhouette_scores: list,
                 k_range: list, dist_pct: pd.DataFrame,
                 n_clusters: int):
    """Buat visualisasi komprehensif hasil clustering."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    features = [b for b in FEATURE_BANDS if b in df.columns]

    # --- 1. PCA biplot + cluster coloring ---
    scaler = RobustScaler()
    X_s = scaler.fit_transform(df[features].values)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_s[:20_000])  # sub-sample untuk speed
    labels_sub = df['cluster_ranked'].values[:20_000]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PCA scatter
    ax = axes[0]
    for c, meta in CLUSTER_LABELS.items():
        if c >= n_clusters:
            continue
        mask = labels_sub == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=meta['color'], alpha=0.3, s=5, label=meta['name'])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA — Distribusi Cluster Spektral", fontweight='bold')
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)

    # Silhouette vs K
    ax = axes[1]
    ax.plot(k_range, silhouette_scores, 'o-', color='#0288D1', linewidth=2)
    ax.set_xlabel("Jumlah Cluster (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Pemilihan K Optimal", fontweight='bold')
    ax.grid(True, alpha=0.3)
    best_k_idx = np.argmax(silhouette_scores)
    ax.axvline(k_range[best_k_idx], color='red', linestyle='--', alpha=0.7,
               label=f"K optimal={k_range[best_k_idx]}")
    ax.legend()

    # Temporal distribution stacked bar
    ax = axes[2]
    colors_stack = [CLUSTER_LABELS[i]['color'] for i in range(n_clusters)
                    if CLUSTER_LABELS[i]['name'] in dist_pct.columns]
    cols_ordered = [CLUSTER_LABELS[i]['name'] for i in range(n_clusters)
                    if CLUSTER_LABELS[i]['name'] in dist_pct.columns]
    dist_pct[cols_ordered].plot(
        kind='bar', stacked=True, ax=ax,
        color=colors_stack, edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Persentase Piksel (%)")
    ax.set_title("Distribusi Zona Kualitas Air per Tahun", fontweight='bold')
    ax.set_xticklabels(dist_pct.index, rotation=0)
    ax.legend(title="Zona", loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    fig.suptitle("Klasifikasi Zona Kualitas Air — K-Means Unsupervised\n"
                 "Pesisir Jakarta & Banten",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, 'ml_kmeans_clustering.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\n   📊 Visualisasi disimpan: {path}")
    plt.close()


def main():
    print("\n🤖 MACHINE LEARNING (Unsupervised) — Zona Kualitas Air\n")

    authenticate_gee()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Load atau sample data
    csv_path = os.path.join(RESULTS_DIR, 'ml_spectral_samples.csv')
    if os.path.exists(csv_path):
        print(f"📂 Data spektral ditemukan: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = collect_spectral_samples(n_points=5000)

    print(f"📋 Dataset: {len(df)} sampel, kolom: {list(df.columns)}\n")

    # Tentukan K optimal
    features = [b for b in FEATURE_BANDS if b in df.columns]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[features].values)
    best_k, silhouette_scores, k_range = find_optimal_k(X_scaled)

    # Run clustering
    df_clustered, pipeline, cluster_stats = run_kmeans_clustering(df, best_k)

    # Analisis temporal
    dist_pct = analyze_temporal_distribution(df_clustered)

    # Simpan hasil
    results = {
        'method': 'K-Means Unsupervised Clustering',
        'n_clusters': best_k,
        'features_used': features,
        'cluster_statistics': cluster_stats,
        'temporal_distribution_pct': dist_pct.to_dict(),
        'note': (
            'Supervised prediction of satellite-derived indices from their own '
            'input bands (circular logic) has been replaced with unsupervised '
            'spatial clustering. For supervised learning, provide in-situ '
            'field measurements as the target variable (y).'
        )
    }

    out_path = os.path.join(RESULTS_DIR, 'ml_clustering_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Hasil clustering: {out_path}")

    # Simpan tabel distribusi temporal
    dist_path = os.path.join(RESULTS_DIR, 'cluster_temporal_distribution.csv')
    dist_pct.to_csv(dist_path)
    print(f"💾 Distribusi temporal: {dist_path}")

    # Plot
    plot_results(df_clustered, silhouette_scores, k_range, dist_pct, best_k)

    print(f"\n✅ MACHINE LEARNING SELESAI!")
    print(f"   📊 Hasil: {out_path}")
    print(f"\n   ⚠️  CATATAN UNTUK PAPER:")
    print(f"   Jika data in-situ tersedia, ganti cluster_label dengan")
    print(f"   nilai lapangan sebagai target (y) untuk supervised learning.")


if __name__ == '__main__':
    main()
