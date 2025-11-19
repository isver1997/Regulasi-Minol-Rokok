import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv("regulasi.csv")

st.title("Analisis Regulasi Minol vs Tembakau")

summary = df.groupby(["domain","sector"])["intensity_score"].mean().reset_index()

st.subheader("Bar Chart Intensitas")
fig, ax = plt.subplots()
sns.barplot(data=summary, x="domain", y="intensity_score", hue="sector", ax=ax)
st.pyplot(fig)

st.subheader("Heatmap Intensitas")
pivot = summary.pivot(index="domain", columns="sector", values="intensity_score").fillna(0)
fig2, ax2 = plt.subplots()
sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax2)
st.pyplot(fig2)

st.subheader("Gap Regulasi Eksklusif Minol")
presence = df.groupby(["sector","domain"])["presence"].sum().reset_index()
pivot_presence = presence.pivot(index="domain", columns="sector", values="presence").fillna(0)
exclusive_domains = pivot_presence.index[(pivot_presence["Minol"]>0)&(pivot_presence["Tembakau"]==0)].tolist()
gap_table = df[(df["sector"]=="Minol") & (df["domain"].isin(exclusive_domains))]
st.dataframe(gap_table[["domain","regulasi","level","intensity_score"]])