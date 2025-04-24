# utils.py
import pandas as pd
import numpy as np
import re
import time
import gc
from contextlib import contextmanager
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).min:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).min:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

def feature_engineering(application_train_short, bureau_short, previous_application_short,
                        pos_cash_balance, installments_payments, credit_balance):
    with timer("Processing application_train_short"):
        df = application_train_short.copy()
        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    with timer("Processing bureau_short"):
        bureau_short, _ = one_hot_encoder(bureau_short)
        bureau_agg = bureau_short.groupby("SK_ID_CURR").agg({'CREDIT_DAY_OVERDUE': 'max', 'AMT_CREDIT_SUM': 'sum'})
        bureau_agg.columns = [f"BURO_{col}" for col in bureau_agg.columns]
        df = df.merge(bureau_agg, how='left', on='SK_ID_CURR')
        del bureau_short, bureau_agg
        gc.collect()

    with timer("Processing previous applications"):
        previous_application_short, _ = one_hot_encoder(previous_application_short)
        prev_agg = previous_application_short.groupby("SK_ID_CURR").agg({'AMT_APPLICATION': 'mean', 'AMT_CREDIT': 'mean'})
        prev_agg.columns = [f"PREV_{col}" for col in prev_agg.columns]
        df = df.merge(prev_agg, how='left', on='SK_ID_CURR')
        del previous_application_short, prev_agg
        gc.collect()

    with timer("Processing POS-CASH balance"):
        pos_cash_balance, _ = one_hot_encoder(pos_cash_balance)
        pos_agg = pos_cash_balance.groupby("SK_ID_CURR").agg({'SK_DPD': 'max', 'CNT_INSTALMENT_FUTURE': 'mean'})
        pos_agg.columns = [f"POS_{col}" for col in pos_agg.columns]
        df = df.merge(pos_agg, how='left', on='SK_ID_CURR')
        del pos_cash_balance, pos_agg
        gc.collect()

    with timer("Processing installments payments"):
        installments_payments, _ = one_hot_encoder(installments_payments)
        ins_agg = installments_payments.groupby("SK_ID_CURR").agg({'AMT_PAYMENT': 'sum', 'DAYS_INSTALMENT': 'mean'})
        ins_agg.columns = [f"INSTAL_{col}" for col in ins_agg.columns]
        df = df.merge(ins_agg, how='left', on='SK_ID_CURR')
        del installments_payments, ins_agg
        gc.collect()

    with timer("Processing credit card balance"):
        credit_balance, _ = one_hot_encoder(credit_balance)
        credit_agg = credit_balance.groupby("SK_ID_CURR").agg({'AMT_BALANCE': 'mean', 'AMT_DRAWINGS_CURRENT': 'sum'})
        credit_agg.columns = [f"CC_{col}" for col in credit_agg.columns]
        df = df.merge(credit_agg, how='left', on='SK_ID_CURR')
        del credit_balance, credit_agg
        gc.collect()

    df = reduce_memory_usage(df)
    return df

def clean_column_names(cols):
    return [re.sub(r'_+', '_', re.sub(r'[^A-Za-z0-9_]', '_', col)).strip('_') for col in cols]
