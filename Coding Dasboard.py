import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================================================
# KONFIGURASI
# ======================================================
st.set_page_config(page_title="Dashboard Analisis 20 Soal", layout="wide")
st.title("📊 DASHBOARD ANALISIS 20 SOAL")
st.caption("Tugas Mata Kuliah Fisika Komputasi")

# ======================================================
# UPLOAD DATA
# ======================================================
uploaded_file = st.file_uploader("Upload Data Excel (50 Responden x 20 Soal)", type=["xlsx"])

if uploaded_file is None:
    st.warning("Silakan upload file terlebih dahulu.")
    st.stop()

df = pd.read_excel(uploaded_file)
indikator = df.select_dtypes(include=[np.number])

st.success(f"✅ Data siap! {len(df)} responden, {indikator.shape[1]} soal")

# ======================================================
# 1️⃣ STATISTIK DESKRIPTIF
# ======================================================
st.header("1️⃣ STATISTIK DESKRIPTIF")

statistik = indikator.describe().T
st.subheader("📊 Tabel Statistik")
st.dataframe(statistik)

st.subheader("📈 Grafik Rata-rata")
mean_per_soal = indikator.mean()

fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.bar(mean_per_soal.index, mean_per_soal.values)
plt.xticks(rotation=45)
st.pyplot(fig1)

# ======================================================
# 2️⃣ ANALISIS PER SOAL
# ======================================================
st.header("2️⃣ ANALISIS PER SOAL")

soal_pilih = st.selectbox("Pilih Soal:", indikator.columns)

data_soal = indikator[soal_pilih]

st.write("Distribusi Skor Statistik")
st.write(f"""
- Rata-rata: {data_soal.mean():.2f}
- Median: {data_soal.median():.2f}
- Std Deviasi: {data_soal.std():.2f}
- Minimum: {data_soal.min()}
- Maximum: {data_soal.max()}
""")

fig2, ax2 = plt.subplots()
ax2.hist(data_soal, bins=5)
ax2.set_title(f"Distribusi {soal_pilih}")
st.pyplot(fig2)

# ======================================================
# 3️⃣ SOAL TERBAIK & TERBURUK
# ======================================================
st.header("3️⃣ SOAL TERBAIK & TERBURUK")

top5 = mean_per_soal.sort_values(ascending=False).head(5)
bottom5 = mean_per_soal.sort_values().head(5)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 5 Soal Terbaik")
    st.write(top5)

with col2:
    st.subheader("📉 5 Soal Terburuk")
    st.write(bottom5)

# ======================================================
# 4️⃣ ANALISIS GAP
# ======================================================
st.header("4️⃣ ANALISIS GAP")

skor_maks = indikator.max().max()
gap = skor_maks - mean_per_soal

prioritas = gap.idxmax()

st.write(f"🎯 Fokus Perbaikan: **{prioritas}**")

fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.bar(gap.index, gap.values)
plt.xticks(rotation=45)
st.pyplot(fig3)

# ======================================================
# 5️⃣ SEGMENTASI SISWA
# ======================================================
st.header("5️⃣ SEGMENTASI SISWA")

jumlah_cluster = st.slider("Jumlah Kelompok:", 2, 5, 3)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(indikator)

kmeans = KMeans(n_clusters=jumlah_cluster, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_mean = df.groupby("Cluster")[indikator.columns].mean()
st.subheader("📊 Profil per Kelompok")
st.dataframe(cluster_mean)

fig4, ax4 = plt.subplots()
df["Cluster"].value_counts().plot(kind="bar", ax=ax4)
st.pyplot(fig4)

# ======================================================
# 6️⃣ KESIMPULAN
# ======================================================
st.header("6️⃣ KESIMPULAN")

soal_terbaik = mean_per_soal.idxmax()
soal_terburuk = mean_per_soal.idxmin()
rata_total = mean_per_soal.mean()

st.write(f"""
Soal Terbaik: **{soal_terbaik} ({mean_per_soal.max():.2f})**  
Soal Terburuk: **{soal_terburuk} ({mean_per_soal.min():.2f})**  
Rata-rata Total: **{rata_total:.2f}**
""")

st.subheader("📋 REKOMENDASI")

for i, soal in enumerate(bottom5.index, 1):
    st.write(f"{i}. {soal} (rata-rata: {bottom5[soal]:.2f}) - Perlu ditingkatkan")

st.success("✅ ANALISIS SELESAI!")
