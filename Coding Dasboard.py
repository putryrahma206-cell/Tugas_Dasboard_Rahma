import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis Butir", layout="wide")
st.title("🎓 Dashboard Analisis Hasil Belajar")
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

fig1, ax1 = plt.subplots()
ax1.hist(df["Total"], bins=10)
ax1.set_xlabel("Skor Total")
ax1.set_ylabel("Frekuensi")
st.pyplot(fig1)

mean_score = df["Total"].mean()
std_score = df["Total"].std()

st.subheader("📌 Kesimpulan Distribusi")
if std_score < 2:
    st.info("Sebaran nilai relatif homogen.")
else:
    st.info("Sebaran nilai cukup bervariasi antar siswa.")

# ==========================================================
# TINGKAT KESULITAN
# ==========================================================
st.divider()
st.subheader("Tingkat Kesulitan Soal")

difficulty = indikator.mean()

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.bar(difficulty.index, difficulty.values)
ax2.set_ylabel("Proporsi Benar")
ax2.set_title("Indeks Kesulitan (p)")
plt.xticks(rotation=45)
st.pyplot(fig2)

jumlah_sulit = sum(difficulty < 0.3)
jumlah_mudah = sum(difficulty > 0.7)

st.subheader("📌 Kesimpulan Tingkat Kesulitan")
st.write(f"Soal Sulit: {jumlah_sulit} | Soal Mudah: {jumlah_mudah}")

# ==========================================================
# DAYA PEMBEDA
# ==========================================================
st.divider()
st.subheader("Daya Pembeda Soal")

df_sorted = df.sort_values("Total", ascending=False)
n = int(0.27 * len(df))

atas = df_sorted.head(n)
bawah = df_sorted.tail(n)

discrimination = {
    col: atas[col].mean() - bawah[col].mean()
    for col in indikator.columns
}

df_disc = pd.DataFrame({
    "Soal": discrimination.keys(),
    "Daya Pembeda": discrimination.values()
})

fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.bar(df_disc["Soal"], df_disc["Daya Pembeda"])
ax3.set_ylabel("Daya Pembeda (D)")
plt.xticks(rotation=45)
st.pyplot(fig3)

buruk = sum(df_disc["Daya Pembeda"] < 0.2)

st.subheader("📌 Kesimpulan Daya Pembeda")
if buruk > 0:
    st.warning("Terdapat butir dengan daya pembeda rendah.")
else:
    st.success("Mayoritas butir memiliki daya pembeda baik.")

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

df_cluster = pd.DataFrame({
    "Cluster": cluster_mean.index,
    "Rata-rata Skor": cluster_mean.values
})

st.dataframe(df_cluster)

# ==========================================================
# RINGKASAN AKHIR VARIATIF
# ==========================================================
st.divider()
st.header("📄 Ringkasan Analisis Otomatis")

prop_sulit = jumlah_sulit / indikator.shape[1]
prop_buruk = buruk / indikator.shape[1]

if mean_score > indikator.shape[1] * 0.75:
    performa = "tinggi"
elif mean_score > indikator.shape[1] * 0.5:
    performa = "sedang"
else:
    performa = "rendah"

if prop_sulit > 0.4:
    kesulitan_umum = "Tes cenderung menantang dengan dominasi butir sulit."
elif prop_sulit < 0.2:
    kesulitan_umum = "Tes relatif mudah bagi sebagian besar siswa."
else:
    kesulitan_umum = "Komposisi tingkat kesulitan cukup proporsional."

if prop_buruk > 0.3:
    kualitas = "Sejumlah butir perlu direvisi karena daya pembeda rendah."
elif prop_buruk > 0:
    kualitas = "Sebagian kecil butir dapat diperbaiki."
else:
    kualitas = "Butir soal secara umum memiliki kualitas baik."

ringkasan = f"""
Performa siswa secara umum berada pada kategori {performa}.
Rata-rata skor kelas adalah {mean_score:.2f} dengan standar deviasi {std_score:.2f}.
{kesulitan_umum}
{jumlah_sulit} butir tergolong sulit.
{kualitas}
"""

st.write(ringkasan)

st.markdown("### 🧾 Rekomendasi")
if prop_buruk > 0:
    st.write("Disarankan merevisi butir dengan daya pembeda rendah dan melakukan uji coba ulang.")
else:
    st.write("Tes dapat digunakan kembali dengan mempertahankan struktur butir yang ada.")

st.success("✅ Dashboard siap digunakan tanpa error library.")
