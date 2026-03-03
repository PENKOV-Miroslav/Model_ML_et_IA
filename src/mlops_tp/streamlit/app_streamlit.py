import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Analyse Exploratoire (EDA)",
    layout="wide",
    page_icon="📊"
)

# ==========================================
# STYLE
# ==========================================
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stMetric {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# TITRE
# ==========================================
st.title("📊 Analyse Exploratoire des Données")
st.markdown("---")

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_PATH = BASE_DIR / "data" / "university_query_train.csv"
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error("❌ Fichier non trouvé. Vérifie le chemin du CSV.")
        return None

df = load_data()

if df is None:
    st.stop()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Navigation")

section = st.sidebar.radio(
    "Choisir une section",
    [
        "Vue Générale",
        "Analyse Univariée",
        "Analyse Bivariée",
        "Corrélations",
        "Analyse par Cible"
    ]
)

# ==========================================
# VUE GENERALE
# ==========================================
if section == "Vue Générale":

    st.header("📌 Aperçu du Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de lignes", df.shape[0])
    col2.metric("Nombre de colonnes", df.shape[1])
    col3.metric("Valeurs manquantes", int(df.isnull().sum().sum()))

    st.markdown("### 🔎 Aperçu des données")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### 📋 Types des variables")
    types_df = pd.DataFrame(df.dtypes, columns=["Type"])
    st.dataframe(types_df, use_container_width=True)

    st.markdown("### 📊 Statistiques descriptives")
    st.dataframe(df.describe(include="all"), use_container_width=True)

# ==========================================
# ANALYSE UNIVARIEE
# ==========================================
elif section == "Analyse Univariée":

    st.header("📊 Analyse Univariée")

    variable = st.selectbox("Choisir une variable", df.columns)

    fig, ax = plt.subplots(figsize=(8, 4))

    if pd.api.types.is_numeric_dtype(df[variable]):

        sns.histplot(x=df[variable].dropna().astype(float), kde=True, ax=ax)
        ax.set_title(f"Distribution de {variable}")
        st.pyplot(fig)

        st.markdown("### 📈 Statistiques")
        st.write(df[variable].describe())

    else:

        sns.countplot(x=df[variable], ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f"Fréquences de {variable}")
        st.pyplot(fig)

        st.markdown("### 📊 Fréquences")
        st.write(df[variable].value_counts())

# ==========================================
# ANALYSE BIVARIEE
# ==========================================
elif section == "Analyse Bivariée":

    st.header("📈 Analyse Bivariée")

    col1, col2 = st.columns(2)
    var1 = col1.selectbox("Variable 1", df.columns)
    var2 = col2.selectbox("Variable 2", df.columns)

    fig, ax = plt.subplots(figsize=(6, 4))

    if pd.api.types.is_numeric_dtype(df[var1]) and pd.api.types.is_numeric_dtype(df[var2]):

        sns.scatterplot(x=df[var1], y=df[var2], ax=ax)
        ax.set_title(f"{var1} vs {var2}")
        st.pyplot(fig)

        corr = df[[var1, var2]].corr().iloc[0, 1]
        st.metric("Corrélation (Pearson)", f"{corr:.3f}")

    elif not pd.api.types.is_numeric_dtype(df[var1]) and not pd.api.types.is_numeric_dtype(df[var2]):

        st.markdown("### 📊 Tableau croisé")
        cross_tab = pd.crosstab(df[var1], df[var2])
        st.dataframe(cross_tab, use_container_width=True)

    else:

        sns.boxplot(x=df[var1], y=df[var2], ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f"{var1} vs {var2}")
        st.pyplot(fig)

# ==========================================
# CORRELATIONS
# ==========================================
elif section == "Corrélations":

    st.header("🔗 Matrice de Corrélation")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        st.warning("Pas assez de variables numériques pour calculer la corrélation.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )
        st.pyplot(fig)

# ==========================================
# ANALYSE PAR CIBLE
# ==========================================
elif section == "Analyse par Cible":

    st.header("🎯 Analyse par Variable Cible")

    target = st.selectbox("Choisir la variable cible", df.columns)

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if col != target:

            st.subheader(f"{col} vs {target}")

            fig, ax = plt.subplots(figsize=(6, 4))

            if pd.api.types.is_numeric_dtype(df[target]):
                sns.scatterplot(x=df[target], y=df[col], ax=ax)
            else:
                sns.boxplot(x=df[target], y=df[col], ax=ax)

            plt.xticks(rotation=45)
            st.pyplot(fig)

st.markdown("---")
st.caption("Application Streamlit - Analyse Exploratoire des Données")