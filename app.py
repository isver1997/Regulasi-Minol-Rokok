import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import re

# --- Konfigurasi halaman ---
st.set_page_config(page_title="Analisis Regulasi Minol vs Tembakau", layout="wide")
st.title("Analisis Regulasi Minol vs Tembakau")

# --- Step 1: Baca CSV ---
df = pd.read_csv("regulasi.csv")

# --- Step 2: Preprocessing ---
# Bobot hierarki level
level_weights = {"UUD":5,"UU":4,"PP/Perpres":3,"Permen":2,"Peraturan Lembaga":1}

# Pastikan tipe data benar
df["presence"] = df["presence"].astype(int)
df["detail"] = df["detail"].astype(int)
df["sanction"] = df["presence"]   # aturanmu: presence = sanction
df["level_score"] = df["level"].map(level_weights).fillna(0).astype(int)

# Skor intensitas
df["intensity_score"] = df["level_score"] + df["detail"] + df["sanction"]

# Ekstraksi tahun (regex diperbaiki agar tidak hanya menangkap "20")
def extract_year(s):
    m = re.findall(r"\b(19\d{2}|20\d{2})\b", str(s))
    return int(m[0]) if m else None
df["year"] = df["regulasi"].apply(extract_year)

# --- Step 3: Filter Sidebar ---
sector_sel = st.sidebar.multiselect("Pilih sektor", sorted(df["sector"].unique()), default=sorted(df["sector"].unique()))
domain_sel = st.sidebar.multiselect("Pilih domain", sorted(df["domain"].unique()), default=sorted(df["domain"].unique()))
df_f = df[(df["sector"].isin(sector_sel)) & (df["domain"].isin(domain_sel))].copy()

st.subheader("Tabel Data")
st.dataframe(df_f, use_container_width=True)

# --- Step 4: Ringkasan Intensitas ---
summary = df_f.groupby(["domain","sector"])["intensity_score"].mean().reset_index()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Bar Chart Intensitas")
    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(data=summary, x="domain", y="intensity_score", hue="sector", ax=ax, palette="viridis")
    ax.set_xlabel("Domain"); ax.set_ylabel("Rata-rata Intensitas")
    plt.xticks(rotation=20)
    st.pyplot(fig)

with col2:
    st.subheader("Heatmap Intensitas")
    pivot_intensity = summary.pivot(index="domain", columns="sector", values="intensity_score").fillna(0)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.heatmap(pivot_intensity, annot=True, cmap="YlGnBu", linewidths=.5, ax=ax2)
    ax2.set_xlabel(""); ax2.set_ylabel("Domain")
    st.pyplot(fig2)

# --- Step 5: Gap Analysis ---
presence_by_sector_domain = df_f.groupby(["sector","domain"])["presence"].sum().reset_index()
pivot_presence = presence_by_sector_domain.pivot(index="domain", columns="sector", values="presence").fillna(0)
exclusive_domains = pivot_presence.index[(pivot_presence.get("Minol",0)>0) & (pivot_presence.get("Tembakau",0)==0)].tolist()
df_f["exclusive_flag"] = ((df_f["sector"]=="Minol") & (df_f["domain"].isin(exclusive_domains)) & (df_f["presence"]==1)).astype(int)

st.subheader("Aturan Eksklusif Minol (tanpa padanan di Tembakau)")
st.dataframe(df_f[df_f["exclusive_flag"]==1][["domain","regulasi","level","intensity_score","year"]].sort_values(["domain","level"], ascending=[True, False]))

st.markdown("> Insight untuk Rekomendasi Kebijakan (Prediktif): Minol lebih ketat di domain supply chain (Distribusi, Standar Mutu, Daring). Sedangkan produk tembakau dominan pada Label/Iklan. Maka melihat pada kondisi tersebut, sebagai dua produk yang dikenai cukai dengan tujuan mengurangi konsumsinya, perlu ada penyeimbangan regulasi agar mencapai keadilan secara tujuan dari diberlakukannya cukai, dengan cara membuat regulasi yang serupa yang diterapkan kepada dua produk yang dikenai cukai tersebut. Meskipun tentu secara yuridis-sosiologis, terdapat parameter yang menyebabkan latar belakangnya regulasi yang berbeda pada dua produk cukai itu, misalnya sebagian besar muslim di Indonesia yang mengharamkan konsumsi minol, dan produk tembakau yang mempengaruhi perekonomian nasional dari pendapatan cukai serta terhadap usaha dan buruh tembakau. Namun, tetap memerlukan kajian normatif yang komprehensif agar produk tembakau sebagai produk cukai yang perlu dikurangi konsumsinya untuk menurunkan tingkat prevalensi perokok di Indonesia dengan pertimbangan public health. Disinilah diperlukan rekomendasi kebijakan yang bersifat prediktif, agar pemerintah menerapkan regulasi yang sama ketatnya atas perlakuan peredaran dan konsumsi dua produk yang dikenai cukai tersebut.")

# --- Step 6: Tren Regulasi per Tahun + Prediksi Sederhana ---
st.subheader("Tren Regulasi per Tahun")

# Hitung jumlah regulasi per tahun per sektor
trend = df_f.groupby(["year","sector"])["regulasi"].count().reset_index()

fig3, ax3 = plt.subplots(figsize=(8,5))
sns.lineplot(data=trend, x="year", y="regulasi", hue="sector", marker="o", ax=ax3)
ax3.set_title("Jumlah Regulasi per Tahun (Minol vs Tembakau)")
ax3.set_xlabel("Tahun")
ax3.set_ylabel("Jumlah Regulasi")
st.pyplot(fig3)

# --- Prediksi sederhana: extrapolasi linear ke depan ---
st.subheader("Prediksi Tren Regulasi (Sederhana)")

# Ambil rata-rata pertambahan regulasi per sektor
growth = trend.groupby("sector")["regulasi"].diff().groupby(trend["sector"]).mean().fillna(0)

future_years = [max(df_f["year"].dropna())+2, max(df_f["year"].dropna())+4]
pred_data = []
for sector in trend["sector"].unique():
    last_val = trend[trend["sector"]==sector]["regulasi"].iloc[-1]
    for i, fy in enumerate(future_years, start=1):
        pred_data.append({"year": fy, "sector": sector, "regulasi": last_val + i*growth[sector]})

pred_df = pd.DataFrame(pred_data)

fig4, ax4 = plt.subplots(figsize=(8,5))
sns.lineplot(data=trend, x="year", y="regulasi", hue="sector", marker="o", ax=ax4)
sns.lineplot(data=pred_df, x="year", y="regulasi", hue="sector", marker="x", linestyle="--", ax=ax4)
ax4.set_title("Prediksi Jumlah Regulasi ke Depan")
ax4.set_xlabel("Tahun")
ax4.set_ylabel("Jumlah Regulasi")
st.pyplot(fig4)

st.markdown("> Catatan: Prediksi ini bersifat ilustratif (linear sederhana). Tujuannya menunjukkan arah tren regulasi, bukan hasil forecasting yang presisi. Dari tren terlihat Minol cenderung bertambah aturan teknis, sementara Tembakau stagnan. Rekomendasi kebijakan: memperkuat regulasi distribusi dan promosi daring untuk Tembakau agar seimbang dengan Minol.")
