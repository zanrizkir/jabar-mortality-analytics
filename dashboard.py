import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, desc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# KONFIGURASI HALAMAN STREAMLIT
st.set_page_config(
    page_title="Dashboard Kematian Jabar",
    page_icon="📊",
    layout="wide"
)

# INISIALISASI SPARK (BACKEND)
@st.cache_resource
def create_spark_session():
    return SparkSession.builder \
        .appName("DashboardKematianJabar") \
        .master("local[*]") \
        .getOrCreate()

spark = create_spark_session()

#LOAD & PREPROCESSING DATA (ETL)
@st.cache_resource
def load_data():
    #AKUISISI DATA
    try:
        #Pastikan file dataset.csv ada di folder 'data'
        df = spark.read.csv("data/dataset.csv", header=True, inferSchema=True)
    except Exception as e:
        st.error(f"Gagal membaca dataset: {e}. Pastikan file 'dataset.csv' ada di folder 'data'.")
        return None
    
    #PRAPROSES
    #Hapus Duplikat
    df = df.dropDuplicates()
    
    #Seleksi & Casting Tipe Data
    df_clean = df.select(
        col("tahun"),
        col("nama_kabupaten_kota"),
        col("jenis_kematian"),
        col("penyebab_kematian"),
        col("jumlah_kematian").cast("int").alias("jumlah")
    )
    
    #Hapus data kosong
    df_clean = df_clean.dropna(subset=["jumlah"])
    return df_clean

df_spark = load_data()

#Hentikan program jika data gagal diload
if df_spark is None:
    st.stop()

#VISUALISASI DASHBOARD (FRONTEND)

#STYLING CSS
st.markdown("""
<style>
    .card-big {
        background-color: #ffffff;
        padding: 20px;
        border-left: 6px solid #007bff;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .card-title {
        color: #007bff;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 8px;
    }
    .card-text {
        font-size: 16px;
        color: #333;
        line-height: 1.5;
    }
    .section-header {
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-top: 40px;
        margin-bottom: 20px;
        color: #2c3e50;
        border-bottom: 2px solid #eee;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard Analisis Kematian Jawa Barat")
st.markdown("Sistem Big Data Analytics menggunakan **PySpark** & **Streamlit**")

#SIDEBAR: FILTER & DOWNLOAD
st.sidebar.header("🎛️ Filter Data")

# Ambil list tahun unik
tahun_data = [row['tahun'] for row in df_spark.select('tahun').distinct().sort('tahun').collect()]
filter_options = ["Semua Tahun"] + tahun_data
selected_option = st.sidebar.selectbox("Pilih Rentang Waktu", filter_options)

# Logika Filter
if selected_option == "Semua Tahun":
    df_filtered = df_spark
    judul_suffix = "(Total 2017-2019)"
else:
    df_filtered = df_spark.filter(col("tahun") == selected_option)
    judul_suffix = f"(Tahun {selected_option})"

# Sidebar: Tombol Download
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Unduh Laporan")
if st.sidebar.button("Download Data (CSV)"):
    csv_data = df_filtered.toPandas().to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Klik disini untuk Simpan",
        data=csv_data,
        file_name=f"laporan_kematian_{str(selected_option).replace(' ', '_')}.csv",
        mime="text/csv"
    )

#KPI (METRICS) ---
total_kematian = df_filtered.agg(_sum("jumlah")).collect()[0][0]
total_kasus = df_filtered.count()

if total_kematian is None: total_kematian = 0

col1, col2 = st.columns(2)
col1.metric("💀 Total Kematian (Jiwa)", f"{total_kematian:,}")
col2.metric("📋 Jumlah Record Data", f"{total_kasus:,}")

st.divider()

#GRAFIK UTAMA 
col_grafik1, col_grafik2 = st.columns(2)

with col_grafik1:
    st.subheader(f"Top 10 Penyebab Kematian")
    st.caption(judul_suffix)
    
    df_cause = df_filtered.groupBy("penyebab_kematian") \
        .agg(_sum("jumlah").alias("total")) \
        .orderBy(desc("total")) \
        .limit(10)
    pdf_cause = df_cause.toPandas()

    if not pdf_cause.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=pdf_cause, x="total", y="penyebab_kematian", palette="viridis", ax=ax)
        ax.set_xlabel("Jumlah")
        ax.set_ylabel("")
        st.pyplot(fig)
    else:
        st.warning("Data kosong.")

with col_grafik2:
    st.subheader(f"Top 5 Wilayah")
    st.caption(judul_suffix)
    
    df_city = df_filtered.groupBy("nama_kabupaten_kota") \
        .agg(_sum("jumlah").alias("total")) \
        .orderBy(desc("total")) \
        .limit(5)
    pdf_city = df_city.toPandas()

    if not pdf_city.empty:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.pie(pdf_city['total'], labels=pdf_city['nama_kabupaten_kota'], autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        st.pyplot(fig2)
    else:
        st.warning("Data kosong.")

#TREN LINE CHART (Khusus "Semua Tahun") ---
if selected_option == "Semua Tahun":
    st.subheader("📈 Tren Kematian dari Waktu ke Waktu")
    
    df_trend = df_spark.groupBy("tahun") \
        .agg(_sum("jumlah").alias("total")) \
        .orderBy("tahun")
    pdf_trend = df_trend.toPandas()
    
    pdf_trend['tahun'] = pdf_trend['tahun'].astype(str)

    fig3, ax3 = plt.subplots(figsize=(10, 3))
    sns.lineplot(data=pdf_trend, x="tahun", y="total", marker="o", linewidth=2.5, color="#e74c3c", ax=ax3)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Label angka di titik grafik
    for x, y in zip(pdf_trend['tahun'], pdf_trend['total']):
        ax3.text(x, y, f'{y:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    st.pyplot(fig3)

#BAGIAN TRANSFORMASI (DIKW) - VERSI VERTIKAL
st.markdown('<div class="section-header">🔄 Transformasi Data Menjadi Pengetahuan</div>', unsafe_allow_html=True)

st.markdown("""
<div class="card-big">
    <div class="card-title">1. Data (Mentah)</div>
    <div class="card-text">
        Dataset bersumber dari Open Data Jabar dalam format CSV. 
        Berisi ribuan baris catatan administratif kejadian kematian per kabupaten/kota (2017-2019). 
        Pada tahap ini, data hanya berupa angka dan teks yang belum memiliki makna strategis.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card-big">
    <div class="card-title">2. Informasi (Terolah)</div>
    <div class="card-text">
        Melalui pemrosesan PySpark (Aggregation & Filtering), data mentah diubah menjadi informasi.
        Contoh dashboard di atas menunjukkan: "Penyebab kematian tertinggi adalah X" dan "Kota Bandung memiliki jumlah kasus terbanyak".
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card-big">
    <div class="card-title">3. Pengetahuan (Pola)</div>
    <div class="card-text">
        Dari grafik tren dan wilayah, ditemukan pola bahwa kematian cenderung meningkat di wilayah padat penduduk (Metropolitan).
        Selain itu, jenis penyebab kematian tertentu memiliki korelasi dengan tahun kejadian.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card-big">
    <div class="card-title">4. Kebijakan (Aksi)</div>
    <div class="card-text">
        Berdasarkan pengetahuan ini, Pemerintah Provinsi dapat memprioritaskan alokasi anggaran kesehatan ke 5 wilayah teratas 
        dan meluncurkan program pencegahan spesifik untuk penyebab kematian yang paling dominan (Top 1 Cause).
    </div>
</div>
""", unsafe_allow_html=True)

#BAGIAN KESIMPULAN - VERSI VERTIKAL
st.markdown('<div class="section-header">✅ Kesimpulan & Rekomendasi</div>', unsafe_allow_html=True)

st.info("**1. Pola Sebaran Wilayah**\n\n"
        "Angka kematian terkonsentrasi di kota-kota besar. "
        "Hal ini mengindikasikan perlunya fokus layanan publik ekstra di area padat penduduk tersebut.")

st.success("**2. Faktor Penyebab Utama**\n\n"
        "Penyebab kematian didominasi oleh faktor penyakit tertentu (lihat grafik). "
        "Dinas Kesehatan dapat menggunakan data ini untuk kampanye kesehatan yang lebih terarah.")

st.warning("**3. Analisis Tren Waktu**\n\n"
        "Fluktuasi angka kematian dari 2017 hingga 2019 menjadi indikator efektivitas program kesehatan pemerintah. "
        "Jika tren naik, diperlukan evaluasi kebijakan segera.")

st.error("**4. Rekomendasi Sistem**\n\n"
         "Implementasi Big Data Analytics terbukti mampu mengolah data mentah menjadi insight real-time "
         "yang mendukung pengambilan keputusan berbasis data (Data Driven Decision Making).")
