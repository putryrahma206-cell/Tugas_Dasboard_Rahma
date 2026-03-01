import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis Butir Lengkap", layout="wide")
st.title("🎓 Dashboard Interaktif Analisis Hasil Belajar")
st.markdown("Analisis butir soal berbasis data 0–1 (Benar/Salah)")

# ==========================================================
# UPLOAD FILE
# ==========================================================
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file Excel", type=["xlsx"])

if uploaded_file is None:
    st.warning("Silakan upload file Excel terlebih dahulu.")
    st.stop()

df = pd.read_excel(uploaded_file)
indikator = df.select_dtypes(include=[np.number])

df["Total"] = indikator.sum(axis=1)
df["Rata-rata"] = indikator.mean(axis=1)

# ==========================================================
# KPI
# ==========================================================
col1, col2, col3 = st.columns(3)
col1.metric("👥 Jumlah Siswa", len(df))
col2.metric("📝 Jumlah Soal", indikator.shape[1])
col3.metric("📊 Rata-rata Skor", f"{df['Total'].mean():.2f}")

st.divider()

# ==========================================================
# DISTRIBUSI NILAI
# ==========================================================
st.subheader("Distribusi Skor Siswa")

fig_hist = px.histogram(df, x="Total", nbins=10, title="Histogram Skor")
st.plotly_chart(fig_hist, use_container_width=True)

mean_score = df["Total"].mean()
std_score = df["Total"].std()

st.subheader("📌 Kesimpulan Distribusi")
if std_score < 2:
    st.info("Sebaran nilai relatif homogen.")
else:
    st.info("Sebaran nilai cukup bervariasi.")

# ==========================================================
# TINGKAT KESULITAN
# ==========================================================
st.divider()
st.subheader("Tingkat Kesulitan Soal")

difficulty = indikator.mean()

df_diff = pd.DataFrame({
    "Soal": difficulty.index,
    "Indeks Kesulitan": difficulty.values
})

def kategori_kesulitan(p):
    if p < 0.3:
        return "Sulit"
    elif p <= 0.7:
        return "Sedang"
    else:
        return "Mudah"

df_diff["Kategori"] = df_diff["Indeks Kesulitan"].apply(kategori_kesulitan)

fig_diff = px.bar(
    df_diff,
    x="Soal",
    y="Indeks Kesulitan",
    color="Kategori",
    title="Indeks Kesulitan (p)"
)

st.plotly_chart(fig_diff, use_container_width=True)

jumlah_sulit = sum(difficulty < 0.3)
jumlah_sedang = sum((difficulty >= 0.3) & (difficulty <= 0.7))
jumlah_mudah = sum(difficulty > 0.7)

st.subheader("📌 Kesimpulan Tingkat Kesulitan")
st.write(f"Soal Sulit: {jumlah_sulit} | Sedang: {jumlah_sedang} | Mudah: {jumlah_mudah}")

# ==========================================================
# DAYA PEMBEDA
# ==========================================================
st.divider()
st.subheader("Daya Pembeda Soal")

df_sorted = df.sort_values("Total", ascending=False)
n = int(0.27 * len(df))

atas = df_sorted.head(n)
bawah = df_sorted.tail(n)

discrimination = {}
for col in indikator.columns:
    discrimination[col] = atas[col].mean() - bawah[col].mean()

df_disc = pd.DataFrame({
    "Soal": discrimination.keys(),
    "Daya Pembeda": discrimination.values()
})

def kategori_daya(D):
    if D >= 0.4:
        return "Sangat Baik"
    elif D >= 0.3:
        return "Baik"
    elif D >= 0.2:
        return "Cukup"
    else:
        return "Buruk"

df_disc["Kategori"] = df_disc["Daya Pembeda"].apply(kategori_daya)

fig_disc = px.bar(
    df_disc,
    x="Soal",
    y="Daya Pembeda",
    color="Kategori",
    title="Indeks Daya Pembeda"
)

st.plotly_chart(fig_disc, use_container_width=True)

buruk = sum(df_disc["Daya Pembeda"] < 0.2)

st.subheader("📌 Kesimpulan Daya Pembeda")
if buruk > 0:
    st.warning("Terdapat soal dengan daya pembeda rendah yang perlu direvisi.")
else:
    st.success("Mayoritas soal memiliki daya pembeda baik.")

# ==========================================================
# SEGMENTASI SISWA
# ==========================================================
st.divider()
st.subheader("Segmentasi Kemampuan Siswa")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(indikator)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_mean = df.groupby("Cluster")["Total"].mean().sort_values(ascending=False)

kategori = ["Tinggi", "Sedang", "Rendah"]

df_cluster = pd.DataFrame({
    "Cluster": cluster_mean.index,
    "Rata-rata Skor": cluster_mean.values,
    "Kategori": kategori
})

st.dataframe(df_cluster)

# ==========================================================
# RINGKASAN AKHIR OTOMATIS
# ==========================================================
st.divider()
st.header("📄 Ringkasan Analisis Otomatis")

ringkasan = f"""
Berdasarkan analisis, rata-rata skor siswa adalah {mean_score:.2f}.
Distribusi nilai menunjukkan variasi sebesar {std_score:.2f}.
Sebanyak {jumlah_sulit} soal tergolong sulit dan {jumlah_mudah} soal tergolong mudah.
Terdapat {buruk} soal dengan daya pembeda rendah yang perlu evaluasi lebih lanjut.
Secara umum, tes dapat digunakan dengan beberapa revisi pada butir tertentu.
"""

st.write(ringkasan)

st.success("✅ Dashboard lengkap siap digunakan untuk presentasi atau laporan penelitian.")
