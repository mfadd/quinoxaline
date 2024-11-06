import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error

import streamlit as st
import time

data_awal = pd.read_excel('Kuinoksalin.xlsx')

data_awal = data_awal.astype(float)


Q1 = data_awal['Con (M)'].quantile(0.05)
Q3 = data_awal['Con (M)'].quantile(0.82)
IQR = Q3 - Q1
data_awal = data_awal[~((data_awal['Con (M)'] < (Q1 - 1.5 * IQR)) | (data_awal['Con (M)'] > (Q3 + 1.5 * IQR)))]

data_awal['EHOMO (eV)'] = np.where(data_awal['EHOMO (eV)'] >-5.8, np.median(data_awal['EHOMO (eV)']),data_awal['EHOMO (eV)'])

data_awal['EHOMO (eV)'] = np.where(data_awal['EHOMO (eV)'] <-7.0, np.median(data_awal['EHOMO (eV)']),data_awal['EHOMO (eV)'])

data_awal['µ'] = np.where(data_awal['µ'] >7, np.median(data_awal['µ']),data_awal['µ'])

data_awal['IP'] = np.where(data_awal['IP'] <5.8, np.median(data_awal['IP']),data_awal['IP'])

data_awal['IP'] = np.where(data_awal['IP'] >7, np.median(data_awal['IP']),data_awal['IP'])

data_awal['ΔΝ'] = np.where(data_awal['ΔΝ'] >0.8, np.median(data_awal['ΔΝ']),data_awal['ΔΝ'])

Q1 = data_awal['IE (%)'].quantile(0.35)
Q3 = data_awal['IE (%)'].quantile(0.85)
IQR = Q3 - Q1
data_awal = data_awal[~((data_awal['IE (%)'] < (Q1 - 1.5 * IQR)) | (data_awal['IE (%)'] > (Q3 + 1.5 * IQR)))]

data_awal = data_awal.reset_index(drop=True)

# Inisialisasi dataset dan fitur-target
#data_awal = pd.read_excel('Kuinoksalin.xlsx')
x = data_awal.drop(columns='IE (%)')
y = data_awal['IE (%)']



# Normalisasi
scaler = RobustScaler()
x = scaler.fit_transform(x)

# ============  Regressor  ============================\n",
from xgboost import XGBRegressor

# K-Fold Cross Validation
kfold = KFold(n_splits=7, shuffle=True, random_state=42)
for train, test in kfold.split(x, y):
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]

# Training
gbr = XGBRegressor()
gbr.fit(x_train, y_train)

# ========================================================================================================================================================================================

# STREAMLIT
st.set_page_config(
  page_title = "Quinoxaline Regression"
)

st.title("Quinoxaline Regression")

st.sidebar.header("**User Input** Sidebar")
ConM = st.sidebar.number_input(label="Con (M)", step=0.001, format="%0.3f")
st.sidebar.write("")

TempKelvin = st.sidebar.number_input(label="Temp (K)")
st.sidebar.write("")

TEeV = st.sidebar.number_input(label="TE (eV)", step=0.01, format="%0.2f")
st.sidebar.write("")

EHOMOeV = st.sidebar.number_input(label="EHOMO (eV)", step=0.001, format="%0.3f")
st.sidebar.write("")

ELUMOeV = st.sidebar.number_input(label="ELUMO (eV)", step=0.001, format="%0.3f")
st.sidebar.write("")

deltaΕeV = st.sidebar.number_input(label="ΔΕ (eV)", step=0.001, format="%0.3f")
st.sidebar.write("")

µ = st.sidebar.number_input(label="µ", step=0.001, format="%0.3f")
st.sidebar.write("")

iP = st.sidebar.number_input(label="IP", step=0.001, format="%0.3f")
st.sidebar.write("")

eA = st.sidebar.number_input(label="EA", step=0.001, format="%0.3f")
st.sidebar.write("")

X = st.sidebar.number_input(label="X", step=0.001, format="%0.3f")
st.sidebar.write("")

η = st.sidebar.number_input(label="η", step=0.001, format="%0.3f")
st.sidebar.write("")

σ = st.sidebar.number_input(label="σ", step=0.001, format="%0.3f")
st.sidebar.write("")

deltaΝ = st.sidebar.number_input(label="ΔΝ", step=0.001, format="%0.3f")
st.sidebar.write("")

data = {
  'Con (M)': ConM,   
  'T (K)': TempKelvin,      
  'TE (eV)': TEeV,    
  'EHOMO (eV)': EHOMOeV, 
  'ELUMO (eV)': ELUMOeV,
  'ΔΕ (eV)': deltaΕeV,
  'µ': µ,           
  'IP': iP,        
  'EA': eA,         
  'Χ': X,          
  'η': η,         
  'σ': σ,          
  'ΔΝ': deltaΝ
}

preview_df = pd.DataFrame(data, index=['input'])

st.header("User Input as DataFrame")
st.write("")
st.dataframe(preview_df.iloc[:, :6])
st.write("")
st.dataframe(preview_df.iloc[:, 6:])
st.write("")


predict_btn = st.button("**Predict**", type="primary")
prediction = 0
st.write("")
if predict_btn:
  inputs = [[ConM, TempKelvin, TEeV, EHOMOeV, ELUMOeV, deltaΕeV, µ, iP, eA, X, η, σ, deltaΝ]]
  prediction = gbr.predict(inputs)[0]

  bar = st.progress(0)
  status_text = st.empty()

  for i in range(1, 101):
    status_text.text(f"{i}% complete")
    bar.progress(i)
    time.sleep(0.01)
    if i == 100:
      time.sleep(1)
      status_text.empty()
      bar.empty()

st.write("")
st.write("")
st.subheader("Prediction:")
st.subheader(f"{prediction} %")



