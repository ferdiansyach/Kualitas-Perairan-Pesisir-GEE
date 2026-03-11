"""
01 - Setup Environment & Autentikasi
=====================================
Script ini memverifikasi instalasi dependensi
dan mengautentikasi Google Earth Engine.

Jalankan script ini PERTAMA sebelum script lainnya.

Usage:
    python 01_setup_environment.py
"""

import subprocess
import sys


def install_requirements():
    """Install dependensi dari requirements.txt."""
    print("=" * 60)
    print("📦 INSTALASI DEPENDENSI")
    print("=" * 60)
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt',
            '--quiet'
        ])
        print("✅ Semua dependensi berhasil diinstal!\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error saat install: {e}")
        print("   Coba jalankan manual: pip install -r requirements.txt")
        sys.exit(1)


def verify_imports():
    """Verifikasi semua modul penting bisa di-import."""
    print("=" * 60)
    print("🔍 VERIFIKASI MODUL")
    print("=" * 60)
    
    modules = {
        'ee': 'Google Earth Engine API',
        'geemap': 'GeeMap (Interactive Mapping)',
        'rasterio': 'Rasterio (GeoTIFF I/O)',
        'geopandas': 'GeoPandas (Geospatial Data)',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'plotly': 'Plotly',
        'folium': 'Folium (Interactive Maps)',
        'sklearn': 'Scikit-Learn',
        'scipy': 'SciPy',
        'streamlit': 'Streamlit',
    }
    
    all_ok = True
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} — TIDAK TERINSTAL")
            all_ok = False
    
    if all_ok:
        print("\n✅ Semua modul tersedia!\n")
    else:
        print("\n⚠️  Beberapa modul belum terinstal. Jalankan:")
        print("    pip install -r requirements.txt\n")
    
    return all_ok


def authenticate_gee():
    """Autentikasi dan test koneksi ke Google Earth Engine."""
    print("=" * 60)
    print("🔐 AUTENTIKASI GOOGLE EARTH ENGINE")
    print("=" * 60)
    
    import ee
    
    try:
        ee.Initialize()
        print("✅ Sudah terautentikasi sebelumnya.\n")
    except Exception:
        print("📋 Langkah autentikasi:")
        print("   1. Browser akan terbuka untuk login Google")
        print("   2. Izinkan akses Earth Engine")
        print("   3. Copy-paste kode verifikasi\n")
        
        ee.Authenticate()
        
        project_id = input("Masukkan Google Cloud Project ID Anda: ").strip()
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        
        print("✅ Autentikasi berhasil!\n")
    
    # Test koneksi
    print("🧪 Testing koneksi ke Earth Engine...")
    try:
        image = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20230101T031109_20230101T031451_T48MYT')
        info = image.getInfo()
        print(f"   ✅ Koneksi OK! Berhasil mengakses katalog Sentinel-2.")
        print(f"   📊 Jumlah band: {len(info['bands'])}\n")
    except Exception as e:
        print(f"   ⚠️  Test koneksi gagal: {e}")
        print("   Pastikan Earth Engine API sudah diaktifkan di Google Cloud Console.\n")


def create_directories():
    """Buat struktur direktori proyek."""
    print("=" * 60)
    print("📁 MEMBUAT STRUKTUR DIREKTORI")
    print("=" * 60)
    
    import os
    
    dirs = [
        'data/raw',
        'data/processed',
        'data/results',
        'output/maps',
        'output/charts',
        'output/reports',
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"   📂 {d}/")
    
    print("\n✅ Struktur direktori siap!\n")


def main():
    print("\n" + "🌊" * 30)
    print("  SETUP ENVIRONMENT — Monitoring Perairan Pesisir")
    print("  Jakarta & Banten")
    print("🌊" * 30 + "\n")
    
    # Step 1: Install dependensi
    install_requirements()
    
    # Step 2: Verifikasi imports
    verify_imports()
    
    # Step 3: Buat direktori
    create_directories()
    
    # Step 4: Autentikasi GEE
    authenticate_gee()
    
    print("=" * 60)
    print("🎉 SETUP SELESAI!")
    print("=" * 60)
    print("Langkah selanjutnya:")
    print("  → Jalankan: python 02_cloud_processing.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
