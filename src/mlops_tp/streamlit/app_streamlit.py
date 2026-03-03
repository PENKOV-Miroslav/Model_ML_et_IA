import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analyse Exploratoire", layout="wide")

# =========================
# STYLE
# =========================
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Analyse Exploratoire des Données (EDA)")
st.markdown("---")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    return df

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Paramètres")

section = st.sidebar.radio(
    "Choisir une section",
    [
        "Vue Générale",
        "Analyse Univariée",
        "Analyse Bivariée",
        "Corrélations",
        "Analyse Cible"
    ]
)

# =========================
# VUE GENERALE
# =========================
if section == "Vue Générale":
    st.header("📌 Aperçu du Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de lignes", df.shape[0])
    col2.metric("Nombre de colonnes", df.shape[1])
    col3.metric("Valeurs manquantes", df.isnull().sum().sum())

    st.subheader("🔎 Aperçu des données")
    st.dataframe(df.head())

    st.subheader("📋 Types de variables")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Type"]))

    st.subheader("📊 Statistiques descriptives")
    st.dataframe(df.describe())

# =========================
# ANALYSE UNIVARIEE
# =========================
elif section == "Analyse Univariée":

    st.header("📊 Analyse Univariée")

    variable = st.selectbox("Choisir une variable", df.columns)

    fig, ax = plt.subplots()

    if df[variable].dtype in ['int64', 'float64']:
        sns.histplot(df[variable], kde=True, ax=ax)
        st.pyplot(fig)

        st.write("### 📈 Statistiques")
        st.write(df[variable].describe())

    else:
        sns.countplot(x=df[variable], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.write("### 📊 Fréquences")
        st.write(df[variable].value_counts())

# =========================
# ANALYSE BIVARIEE
# =========================
elif section == "Analyse Bivariée":

    st.header("📈 Analyse Bivariée")

    col1, col2 = st.columns(2)

    var1 = col1.selectbox("Variable 1", df.columns)
    var2 = col2.selectbox("Variable 2", df.columns)

    fig, ax = plt.subplots()

    if df[var1].dtype in ['int64', 'float64'] and df[var2].dtype in ['int64', 'float64']:
        sns.scatterplot(x=df[var1], y=df[var2], ax=ax)
        st.pyplot(fig)

        corr = df[[var1, var2]].corr().iloc[0,1]
        st.metric("Corrélation", round(corr, 3))

    else:
        cross_tab = pd.crosstab(df[var1], df[var2])
        st.dataframe(cross_tab)

# =========================
# MATRICE DE CORRELATION
# =========================
elif section == "Corrélations":

    st.header("🔗 Matrice de Corrélation")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =========================
# ANALYSE CIBLE
# =========================
elif section == "Analyse Cible":

    st.header("🎯 Analyse par variable cible")

    target = st.selectbox("Choisir la variable cible", df.columns)

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if col != target:
            st.subheader(f"{col} vs {target}")

            fig, ax = plt.subplots()
            sns.boxplot(x=df[target], y=df[col], ax=ax)
            st.pyplot(fig)