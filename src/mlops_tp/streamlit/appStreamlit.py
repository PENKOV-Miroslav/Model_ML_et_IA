import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="EDA - MLOps",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Analyse Exploratoire Générique")
st.markdown("Application compatible avec n'importe quel fichier CSV.")

# =========================================================
# UPLOAD DATA
# =========================================================
uploaded_file = st.sidebar.file_uploader("📂 Charger un fichier CSV", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is None:
    st.info("Veuillez charger un fichier CSV.")
    st.stop()

df = load_data(uploaded_file)

# =========================================================
# DÉTECTION AUTOMATIQUE DES TYPES
# =========================================================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

# Détection date automatique
for col in df.columns:
    try:
        df[col] = pd.to_datetime(df[col])
        if col not in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != col]
    except:
        pass

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
section = st.sidebar.radio(
    "Navigation",
    [
        "1️⃣ Vue Générale",
        "2️⃣ Qualité des Données",
        "3️⃣ Analyse Univariée",
        "4️⃣ Analyse Bivariée",
        "5️⃣ Corrélations",
        "6️⃣ Analyse Cible (ML)",
        "7️⃣ Rapport MLOps"
    ]
)

# =========================================================
# 1️⃣ VUE GÉNÉRALE
# =========================================================
if section == "1️⃣ Vue Générale":

    col1, col2, col3 = st.columns(3)
    col1.metric("Lignes", df.shape[0])
    col2.metric("Colonnes", df.shape[1])
    col3.metric("Valeurs manquantes", int(df.isna().sum().sum()))

    st.subheader("Aperçu")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Types de variables")
    types_df = pd.DataFrame({
        "Type": df.dtypes,
        "Missing (%)": (df.isna().sum() / len(df) * 100).round(2)
    })
    st.dataframe(types_df)

# =========================================================
# 2️⃣ QUALITÉ DES DONNÉES
# =========================================================
elif section == "2️⃣ Qualité des Données":

    st.subheader("Valeurs manquantes")
    missing = df.isna().sum().sort_values(ascending=False)
    st.dataframe(missing[missing > 0])

    st.subheader("Doublons")
    duplicates = df.duplicated().sum()
    st.metric("Nombre de doublons", duplicates)

    if numeric_cols:
        st.subheader("Détection d'outliers (IQR)")
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
            st.write(f"{col} → {len(outliers)} outliers")

# =========================================================
# 3️⃣ ANALYSE UNIVARIÉE
# =========================================================
elif section == "3️⃣ Analyse Univariée":

    var = st.selectbox("Choisir une variable", df.columns)

    fig, ax = plt.subplots()

    if var in numeric_cols:
        sns.histplot(data=df, x=var, kde=True, ax=ax)
        st.pyplot(fig)
        st.write(df[var].describe())

    else:
        sns.countplot(data=df, x=var, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write(df[var].value_counts())

# =========================================================
# 4️⃣ ANALYSE BIVARIÉE
# =========================================================
elif section == "4️⃣ Analyse Bivariée":

    col1, col2 = st.columns(2)
    var1 = col1.selectbox("Variable 1", df.columns)
    var2 = col2.selectbox("Variable 2", df.columns)

    fig, ax = plt.subplots()

    if var1 in numeric_cols and var2 in numeric_cols:
        sns.scatterplot(x=df[var1], y=df[var2])
    elif var1 in categorical_cols and var2 in numeric_cols:
        sns.boxplot(x=df[var1], y=df[var2])
        plt.xticks(rotation=45)
    else:
        st.write(pd.crosstab(df[var1], df[var2]))
        st.stop()

    st.pyplot(fig)

# =========================================================
# 5️⃣ CORRÉLATIONS
# =========================================================
elif section == "5️⃣ Corrélations":

    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)
    else:
        st.warning("Pas assez de variables numériques.")

# =========================================================
# 6️⃣ ANALYSE CIBLE (ML READY)
# =========================================================
elif section == "6️⃣ Analyse Cible (ML)":

    target = st.selectbox("Choisir la variable cible", df.columns)

    if target in categorical_cols:
        st.subheader("Distribution des classes")
        st.write(df[target].value_counts(normalize=True))

        if df[target].nunique() == 2:
            st.success("Problème détecté : Classification Binaire")
        else:
            st.success("Problème détecté : Classification Multiclasse")

    elif target in numeric_cols:
        st.success("Problème détecté : Régression")
        st.write(df[target].describe())

    st.subheader("Relation avec variables numériques")
    for col in numeric_cols:
        if col != target:
            fig, ax = plt.subplots()
            if target in categorical_cols:
                sns.boxplot(x=df[target], y=df[col])
            else:
                sns.scatterplot(x=df[col], y=df[target])
            st.pyplot(fig)

# =========================================================
# 7️⃣ RAPPORT MLOPS
# =========================================================
elif section == "7️⃣ Rapport MLOps":

    st.subheader("Résumé exploitable pour le README")

    summary = {
        "dataset_shape": df.shape,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "generated_at": datetime.now().isoformat()
    }

    st.json(summary)

    st.download_button(
        "📥 Télécharger run_info.json",
        data=json.dumps(summary, indent=4),
        file_name="run_info.json"
    )

st.markdown("---")
st.caption("EDA générique orientée MLOps")