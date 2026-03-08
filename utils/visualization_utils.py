"""
Visualization Utilities
=======================
Helper functions untuk visualisasi hasil analisis
kualitas perairan pesisir.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os


# ============================================================
# COLOR MAPS untuk parameter kualitas air
# ============================================================

def get_water_quality_cmap(parameter):
    """
    Mendapatkan colormap yang sesuai untuk setiap parameter.
    
    Parameters
    ----------
    parameter : str
        Nama parameter ('NDCI', 'NDTI', 'TSS', 'CDOM', 'Secchi_Depth', 'NDWI')
        
    Returns
    -------
    matplotlib.colors.Colormap
    """
    cmaps = {
        'NDCI': 'YlGn',           # Kuning-Hijau (chlorophyll)
        'NDTI': 'YlOrBr',         # Kuning-Coklat (turbidity)
        'TSS': 'YlOrRd',          # Kuning-Merah (suspended solids)
        'CDOM': 'PuBu',           # Ungu-Biru (dissolved organic)
        'Secchi_Depth': 'Blues',   # Biru (kejernihan)
        'NDWI': 'RdYlBu',        # Merah-Biru (water index)
        'Water_Mask': 'Blues',     # Biru (mask)
    }
    return plt.get_cmap(cmaps.get(parameter, 'viridis'))


def get_parameter_label(parameter):
    """Mendapatkan label deskriptif untuk parameter."""
    labels = {
        'NDCI': 'Chlorophyll-a Index (NDCI)',
        'NDTI': 'Turbidity Index (NDTI)',
        'TSS': 'Total Suspended Solids (mg/L)',
        'CDOM': 'Colored Dissolved Organic Matter (rasio)',
        'Secchi_Depth': 'Kedalaman Secchi (relatif)',
        'NDWI': 'Normalized Difference Water Index',
        'Water_Mask': 'Water Mask',
    }
    return labels.get(parameter, parameter)


# ============================================================
# VISUALISASI PETA
# ============================================================

def plot_single_parameter(data, parameter, title=None, extent=None,
                          output_path=None, figsize=(12, 8)):
    """
    Plot satu parameter kualitas air sebagai peta.
    
    Parameters
    ----------
    data : np.ndarray
        Data indeks kualitas air (2D array)
    parameter : str
        Nama parameter
    title : str, optional
        Judul plot
    extent : list, optional
        [west, east, south, north] dalam derajat
    output_path : str, optional
        Path untuk menyimpan gambar
    figsize : tuple
        Ukuran figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    cmap = get_water_quality_cmap(parameter)
    label = get_parameter_label(parameter)
    
    # Hitung vmin/vmax dari data valid
    valid_data = data[~np.isnan(data)] if np.any(~np.isnan(data)) else data.flatten()
    vmin = np.percentile(valid_data, 2)
    vmax = np.percentile(valid_data, 98)
    
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"💾 Peta disimpan: {output_path}")
    
    plt.show()
    plt.close()


def plot_multi_parameter(indices_dict, year, extent=None,
                         output_path=None, figsize=(18, 12)):
    """
    Plot semua parameter kualitas air dalam satu figure (grid).
    
    Parameters
    ----------
    indices_dict : dict
        Dictionary berisi semua indeks kualitas air
    year : int
        Tahun analisis
    extent : list, optional
        [west, east, south, north]
    output_path : str, optional
        Path output
    figsize : tuple
        Ukuran figure
    """
    params = ['NDCI', 'NDTI', 'TSS', 'CDOM', 'Secchi_Depth', 'NDWI']
    available = [p for p in params if p in indices_dict]
    
    n = len(available)
    cols = 3
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(f'Parameter Kualitas Perairan Pesisir Jakarta & Banten — {year}',
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, param in enumerate(available):
        ax = axes[i]
        data = indices_dict[param]
        cmap = get_water_quality_cmap(param)
        label = get_parameter_label(param)
        
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 2)
            vmax = np.percentile(valid_data, 98)
        else:
            vmin, vmax = 0, 1
        
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, aspect='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, fontsize=8)
        ax.set_title(param, fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=8)
    
    # Sembunyikan axes kosong
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"💾 Peta multi-parameter disimpan: {output_path}")
    
    plt.show()
    plt.close()


def plot_temporal_comparison(data_by_year, parameter, extent=None,
                             output_path=None, figsize=(20, 5)):
    """
    Plot perbandingan temporal satu parameter antar tahun.
    
    Parameters
    ----------
    data_by_year : dict
        {year: np.ndarray} untuk satu parameter
    parameter : str
        Nama parameter
    extent : list, optional
    output_path : str, optional
    """
    years = sorted(data_by_year.keys())
    n = len(years)
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    cmap = get_water_quality_cmap(parameter)
    label = get_parameter_label(parameter)
    
    # Tentukan vmin/vmax global
    all_data = np.concatenate([data_by_year[y].flatten() for y in years])
    valid = all_data[~np.isnan(all_data)]
    if len(valid) > 0:
        vmin = np.percentile(valid, 2)
        vmax = np.percentile(valid, 98)
    else:
        vmin, vmax = 0, 1
    
    for i, year in enumerate(years):
        ax = axes[i]
        im = ax.imshow(data_by_year[year], cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, aspect='auto')
        ax.set_title(f'{year}', fontsize=13, fontweight='bold')
        ax.tick_params(labelsize=8)
    
    # Colorbar di sisi kanan
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(label, fontsize=11)
    
    fig.suptitle(f'Perubahan Temporal {label}\nPesisir Jakarta & Banten',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"💾 Perbandingan temporal disimpan: {output_path}")
    
    plt.show()
    plt.close()


def plot_statistics_summary(stats_df, output_path=None, figsize=(16, 10)):
    """
    Plot ringkasan statistik per tahun sebagai bar chart.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame dengan kolom: Year, Parameter, Mean, Std, Min, Max
    output_path : str, optional
    """
    import pandas as pd
    
    params = stats_df['Parameter'].unique()
    n = len(params)
    cols = 3
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, param in enumerate(params):
        ax = axes[i]
        subset = stats_df[stats_df['Parameter'] == param]
        
        years = subset['Year'].values
        means = subset['Mean'].values
        stds = subset['Std'].values
        
        bars = ax.bar(years.astype(str), means, color=colors[i % len(colors)],
                      alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.errorbar(years.astype(str), means, yerr=stds, fmt='none',
                    capsize=5, color='black', alpha=0.6)
        
        ax.set_title(get_parameter_label(param), fontsize=10, fontweight='bold')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Nilai')
        ax.tick_params(labelsize=9)
    
    for j in range(len(params), len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Ringkasan Statistik Kualitas Perairan per Tahun',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"💾 Ringkasan statistik disimpan: {output_path}")
    
    plt.show()
    plt.close()
