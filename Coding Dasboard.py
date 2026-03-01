import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis Butir Soal", layout="wide")
st.title("🎓 Dashboard Analisis Hasil Belajar & Butir Soal")
st.markdown("Data simulasi 50 siswa – 20 soal")

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
indikator = df.select_dtypes(include=[np.number])

# Hitung nilai total per siswa
df["Total"] = indikator.sum(axis=1)
df["Rata-rata"] = indikator.mean(axis=1)

# ==========================================================
# KPI UTAMA
# ==========================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Jumlah Siswa", len(df))
col2.metric("📝 Jumlah Soal", indikator.shape[1])
col3.metric("📊 Rata-rata Kelas", f"{df['Rata-rata'].mean():.2f}")
col4.metric("🏆 Skor Maksimum", df["Total"].max())

st.divider()

# ==========================================================
# DISTRIBUSI NILAI
# ==========================================================
st.header("Distribusi Nilai Siswa")

fig1, ax1 = plt.subplots()
ax1.hist(df["Total"], bins=10)
ax1.set_xlabel("Skor Total")
ax1.set_ylabel("Frekuensi")
ax1.set_title("Histogram Skor Siswa")
st.pyplot(fig1)

st.divider()

# ==========================================================
# 1️⃣ TINGKAT KESULITAN SOAL (Difficulty Index)
# ==========================================================
st.header("Tingkat Kesulitan Soal")

difficulty = indikator.mean()

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.bar(difficulty.index, difficulty.values)
ax2.set_ylabel("Proporsi Benar")
ax2.set_title("Indeks Kesulitan (p)")
ax2.set_xticklabels(difficulty.index, rotation=45)
st.pyplot(fig2)

st.info(f"Soal tersulit: {difficulty.idxmin()} (p={difficulty.min():.2f})")
st.success(f"Soal termudah: {difficulty.idxmax()} (p={difficulty.max():.2f})")

st.divider()

# ==========================================================
# 2️⃣ DAYA PEMBEDA (Discrimination Index)
# Metode Kelompok Atas–Bawah (27%)
# ==========================================================
st.header("Daya Pembeda Soal")

# Urutkan berdasarkan skor total
df_sorted = df.sort_values("Total", ascending=False)

n = int(0.27 * len(df))
kelompok_atas = df_sorted.head(n)
kelompok_bawah = df_sorted.tail(n)

discrimination = {}

for col in indikator.columns:
    p_atas = kelompok_atas[col].mean()
    p_bawah = kelompok_bawah[col].mean()
    discrimination[col] = p_atas - p_bawah

discrimination = pd.Series(discrimination)

fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.bar(discrimination.index, discrimination.values)
ax3.set_ylabel("Daya Pembeda (D)")
ax3.set_title("Indeks Daya Pembeda")
ax3.set_xticklabels(discrimination.index, rotation=45)
st.pyplot(fig3)

st.info(f"Soal dengan daya pembeda tertinggi: {discrimination.idxmax()} (D={discrimination.max():.2f})")
st.warning(f"Soal dengan daya pembeda rendah: {discrimination.idxmin()} (D={discrimination.min():.2f})")

st.divider()

# ==========================================================
# 3️⃣ SEGMENTASI SISWA
# ==========================================================
st.header("Segmentasi Kemampuan Siswa")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(indikator)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_mean = df.groupby("Cluster")["Total"].mean()
cluster_mean = cluster_mean.sort_values(ascending=False)

kategori = ["Tinggi", "Sedang", "Rendah"]
cluster_summary = pd.DataFrame({
    "Rata-rata Skor": cluster_mean.values,
    "Kategori": kategori
})

st.dataframe(cluster_summary)

st.success("✅ Analisis butir soal selesai – siap untuk interpretasi akademik.")
