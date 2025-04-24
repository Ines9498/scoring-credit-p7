# ✅ Librairies essentielles
import pandas as pd
import numpy as np
import re
import pickle
import time
import gc
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# ✅ Prétraitement & modèle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from contextlib import contextmanager

from utils import feature_engineering, clean_column_names

app = FastAPI()

# Charger les fichiers constants
bureau = pd.read_csv('data/original/bureau.csv')
bureau_balance = pd.read_csv('data/original/bureau_balance.csv')
credit_balance = pd.read_csv('data/original/credit_card_balance.csv')
installments_payments = pd.read_csv('data/original/installments_payments.csv')
previous_application = pd.read_csv('data/original/previous_application.csv')
pos_cash_balance = pd.read_csv('data/original/POS_CASH_balance.csv')

# Supprimer les colonnes avant la fusion
colonnes_a_supprimer1 = ['AMT_ANNUITY', 'AMT_CREDIT_MAX_OVERDUE']
bureau_short = bureau.drop(columns=colonnes_a_supprimer1)

colonnes_a_supprimer2 = ['AMT_DOWN_PAYMENT', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_TERMINATION', 'NAME_TYPE_SUITE', 'NFLAG_INSURED_ON_APPROVAL', 'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED']
previous_application_short = previous_application.drop(columns=colonnes_a_supprimer2)

# Endpoint principal
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Ajout pour debug l'erreur de fichier vide
    print("Nom du fichier reçu :", file.filename)
    content = await file.read()
    print("Taille du fichier reçu :", len(content), "octets")

    import io
    file.file = io.BytesIO(content)

    try:
        application_test = pd.read_csv(file.file)
        print("Colonnes dans le fichier :", application_test.columns.tolist())
    except Exception as e:
        print("Erreur lecture CSV :", e)
        return JSONResponse(content={"error": str(e)}, status_code=400)

    colonnes_a_supprimer3 = ['FLAG_DOCUMENT_19', 'FLAG_EMAIL', 'FLAG_DOCUMENT_4', 'FLAG_CONT_MOBILE', 'LIVE_REGION_NOT_WORK_REGION', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'FLAG_DOCUMENT_7', 'AMT_REQ_CREDIT_BUREAU_DAY', 'FLAG_MOBIL', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_5', 'OWN_CAR_AGE', 'EXT_SOURCE_1', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'AMT_GOODS_PRICE', 'REGION_RATING_CLIENT_W_CITY']
    application_test_short = application_test.drop(columns=colonnes_a_supprimer3)

    df = feature_engineering(application_test_short,
                              bureau_short,
                              previous_application_short,
                              pos_cash_balance,
                              installments_payments,
                              credit_balance)

    colonnes_a_supprimer4 = ['CC_AMT_DRAWINGS_CURRENT', 'CC_AMT_BALANCE']
    df = df.drop(columns=colonnes_a_supprimer4)

    # Regroupement
    df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].replace(['Separated', 'Widow', 'Unknown'], 'Autre')
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace(['Incomplete higher', 'Lower secondary', 'Academic degree'], 'Autre')
    df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].replace(['Unemployed', 'Student', 'Businessman', 'Maternity leave'], 'Autre')
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace(['Cleaning staff', 'Cooking staff', 'Waiters/barmen staff', 'Low-skill Laborers'], 'Travail manuel')
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace(['IT staff', 'HR staff', 'Secretaries', 'Realty agents'], 'Travail bureau')
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace(['Medicine staff', 'Private service staff'], 'Secteur social')

    for col in df.select_dtypes(include=[np.signedinteger]).columns:
        df[col] = df[col].astype('int64')

    for col in df.select_dtypes(include=[np.floating]).columns:
        df[col] = df[col].astype('float64')

    # Imputation
    num_cols = df.select_dtypes(include=[float, int]).columns
    simple_imputer = SimpleImputer(strategy='median')
    df[num_cols] = simple_imputer.fit_transform(df[num_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    categorical_cols = df.select_dtypes(exclude=[float, int]).columns
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    ohe = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    cat_encoded = pd.DataFrame(ohe.fit_transform(df[categorical_cols]), index=df.index)
    cat_encoded.columns = ohe.get_feature_names_out(categorical_cols)
    df = pd.concat([df.drop(columns=categorical_cols), cat_encoded], axis=1)

    colonnes_a_supprimer5 = ['FLAG_EMP_PHONE', 'LIVE_CITY_NOT_WORK_CITY', 'CNT_FAM_MEMBERS', 'DAYS_EMPLOYED_PERC', 'INCOME_CREDIT_PERC', 'INCOME_PER_PERSON', 'PREV_AMT_CREDIT', 'ORGANIZATION_TYPE_XNA', 'NAME_TYPE_SUITE_Unaccompanied', 'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_INCOME_TYPE_Pensioner', 'SK_ID_CURR']
    df = df.drop(columns=[col for col in colonnes_a_supprimer5 if col in df.columns])

    df.columns = clean_column_names(df.columns)

    # Prédiction
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    df_pred = df.drop(columns=["TARGET"]) if "TARGET" in df.columns else df.copy()
    if hasattr(model, "feature_name_"):
        df_pred = df_pred[model.feature_name_]

    pred_proba = model.predict_proba(df_pred)[:, 1]
    pred_label = model.predict(df_pred)

    return JSONResponse(content={"pred_proba": pred_proba.tolist(), "pred_label": pred_label.tolist()})
