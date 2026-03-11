# 🌊 Monitoring Kualitas Perairan Pesisir Jakarta & Banten via Cloud

![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Cloud%20Processing-48CAE4?style=for-the-badge&logo=google-earth)
![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)

Repositori ini berisi kode sumber akademik dan _pipeline_ analisis untuk memantau kualitas dan kondisi perairan pesisir pantai Jakarta & Banten dari rentang tahun **2019 hingga 2025**.

Pendekatan ini mutlak **100% berbasis Cloud** menggunakan konektivitas Google Earth Engine (GEE). Skrip ini membebaskan perangkat komputer (laptop) dari beban memori *download* data raster mentah yang berukuran gigabytes menjadi hanya ekstraksi matriks statistik JSON, CSV, dan Peta PNG saja.

---

## 🌟 Fitur Utama (Executive Dashboard v2.0)

- **📈 Complete 7-Years Time Series**: Data terekstraksi secara historis penuh dari tahun 2019, 2020, 2021, 2022, 2023, 2024, hingga 2025.
- **🗺️ Interactive Split-Map DualLayer**: Evaluasi langsung area pesisir secara visual menggunakan peta folium berbasis _split-slider_ guna membandingkan *Tahun Awal* melawan *Tahun Akhir*.
- **🧮 Uji Signifikansi (Mann-Kendall)**: Analisis *Temporal Tren* non-parametrik menggunakan Mann-Kendall Test dan Sen's Slope yang robust terhadap *outliers* kecil.
- **💡 Executive Summary Otomatis**: Hasil analisis GEE akan diterjemahkan oleh algoritma menjadi Paragraf Naratif yang mudah dipahami *Non-Programmer* atau Pembuat Kebijakan Publik.
- **🤖 K-Means Clustering**: *Machine Learning* tanpa supervisi untuk men-segmentasikan zona kualitas air secara objektif tanpa "Circular Logic".

## 📊 Parameter Deteksi (ESA Sentinel-2 & NASA Landsat-8)

| Indikator Kualitas Air | Algoritma Sensor | Deskripsi Dampak Lingkungan | Standar Mutu |
|---|---|---|---|
| **Suhu (SST)** | Thermal Band 10 | Pemanasan/stres termal pada koloni karang. | Kepmen LH |
| **Klorofil-a (NDCI)** | B5 & B4 (Red Edge) | Pertumbuhan alga fitoplankton (Eutrofikasi). | Kepmen LH |
| **Material Padat (TSS)** | Empiris (Red) | Beban lumpur, menghalangi cahaya matahari. | Kepmen LH |
| **Bahan Organik (CDOM)**| B2 / B3 | Indikasi masuknya pembuangan limbah (limbah domestik). | - |
| **Turbiditas (NDTI)** | Lacaux et al. | Tingkat kekeruhan air visual. | - |
| **Secchi Depth** | B2 / B3 (Log) | Jarak tembus pandang kejernihan ke bawah air laut. | - |

---

## 🚀 Panduan Menjalankan

Langkah-langkah berurutan untuk memproses arsip satelit hingga menayangkannya dalam bentuk Web Dashboard:

```bash
# 1. Setup Environment (Instalasi Modul / Autentikasi Google Earth Engine)
python 01_setup_environment.py

# 2. Ekstraksi Metrik Resolusi Tinggi & Thumbnail dari Server GEE (Full Cloud)
python 02_cloud_processing.py

# 3. Analisis Change Detection, Korelasi Spasial-Temporal & P-Value
python 03_cloud_analysis.py

# 4. Render Layout Grafik & Visualisasi
python 04_visualization.py

# 5. Modul Opsional Machine Learning (Zona Clustering Unsupervised K-Means)
python 06_machine_learning.py

# 6. Jalankan Pusat Dashboard Utama Ke Layar Monitor Anda:
streamlit run 05_dashboard.py
```

## 📁 Struktur Hirarki Script Moduler

```text
├── 01_setup_environment.py      # Autentikasi earthengine-api
├── 02_cloud_processing.py       # Cloud Map & Ekstraktor Statistik
├── 03_cloud_analysis.py         # Temporal Trend (Mann-Kendall)
├── 04_visualization.py          # Generator Chart (Seaborn/Plotly)
├── 05_dashboard.py              # Front-End Streamlit Web
├── 06_machine_learning.py       # K-Means Unsupervised Clustering
├── utils/
│   ├── gee_utils.py             # Fungsi Koneksi & Filter GEE
│   └── water_indices.py         # Rumus Kalkulus Parameter Air
├── data/results/                # File Data JSON & CSV (< 1MB)
└── output/                      # Peta Kecil (PNG) dan Gambar Chart
```

## 📡 Meta-Data Set & Lisensi

- **Resolusi Geospasial:** 10 Meter (Sentinel) & 30 Meter (Landsat Thermal)
- **Topografi:** Teluk Pesisir Jakarta hingga Selat Banten (Indonesia)
- Seluruh kode pemfilteran perairan didesain dengan ketat membedakan daratan dan laut memakai **Water Threshold Index (NDWI > 0.1)**. 

---
*Dikembangkan sepenuhnya untuk kepentingan riset geospasial kelautan.*
