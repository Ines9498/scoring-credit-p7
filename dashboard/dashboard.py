import streamlit as st
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import pickle
import numpy as np

st.set_page_config(page_title="Dashboard Crédit", layout="wide")

st.title("📊 Dashboard Crédit - Prédictions")

# Upload fichier
uploaded_file = st.file_uploader("📂 Uploader votre fichier application_test.csv", type="csv")

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    st.subheader("📄 Aperçu des données uploadées :", divider='rainbow')
    st.dataframe(df_uploaded.head())

    # Sélection client avant la prédiction
    client_id = st.selectbox("👤 Sélectionner un client", df_uploaded["SK_ID_CURR"])
    st.markdown(f"### Client sélectionné : `{client_id}`", unsafe_allow_html=True)

    # Bouton pour lancer la prédiction
    # Bouton pour lancer la prédiction
    if st.button("🚀 Lancer la prédiction"):
        API_URL = "https://scoring-api-i20f.onrender.com/predict/"
        files = {'file': ('application_test.csv', uploaded_file.getvalue())}
        response = requests.post(API_URL, files=files)



        if response.status_code == 200:
            result = response.json()
            pred_proba = result["pred_proba"]
            pred_label = result["pred_label"]

            df_uploaded["Probabilité défaut"] = pred_proba
            df_uploaded["Prédiction"] = pred_label

            st.success("✅ Prédictions terminées")
            st.dataframe(df_uploaded[["SK_ID_CURR", "Probabilité défaut", "Prédiction"]].head())

            # Détails client sélectionné
            client_row = df_uploaded[df_uploaded["SK_ID_CURR"] == client_id]
            st.subheader("📄 Détails du client sélectionné", divider='rainbow')
            st.dataframe(client_row)

            # Chargement modèle pour SHAP
            with open("models/best_model.pkl", "rb") as f:
                model = pickle.load(f)

            # Préparer les données pour SHAP
            df_shap = df_uploaded.drop(columns=["SK_ID_CURR", "Probabilité défaut", "Prédiction"]).copy()
            df_shap = pd.get_dummies(df_shap, dummy_na=True)

            missing_cols = set(model.feature_name_) - set(df_shap.columns)
            for col in missing_cols:
                df_shap[col] = 0
            df_shap = df_shap[model.feature_name_]

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_shap)

            # 🌍 SHAP Global
            st.subheader("🌍 Importance Globale des variables", divider='rainbow')
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, df_shap, show=False)
            st.pyplot(fig)

            # 🔍 SHAP Local : Waterfall seulement
            index_client = df_uploaded.index[df_uploaded["SK_ID_CURR"] == client_id][0]

            st.subheader("🔍 Explication locale pour le client sélectionné", divider='rainbow')

            st.markdown("#### 🌊 Waterfall Plot")
            fig2, ax2 = plt.subplots()
            shap.waterfall_plot(shap.Explanation(values=shap_values[index_client],
                                                 base_values=explainer.expected_value,
                                                 data=df_shap.iloc[index_client],
                                                 feature_names=df_shap.columns.tolist()))
            st.pyplot(fig2)

            # ℹ️ Infos techniques API
            st.sidebar.subheader("ℹ️ Infos API / Debug")
            st.sidebar.write(f"Statut API : {response.status_code}")
            st.sidebar.write(f"Client ID : {client_id}")
            st.sidebar.write(f"Probabilité défaut : {client_row['Probabilité défaut'].values[0]:.2f}")
            st.sidebar.write(f"Prédiction : {'Crédit Accordé' if client_row['Prédiction'].values[0]==0 else 'Crédit Refusé'}")

        else:
            st.error(f"❌ Erreur API : {response.status_code}")
