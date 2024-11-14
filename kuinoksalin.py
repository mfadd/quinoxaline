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


# Inisialisasi halaman default di session state
if "halaman" not in st.session_state:
    st.session_state.halaman = "Halaman Utama"

# Membuat tombol di sidebar dan mengubah halaman saat tombol diklik
with st.sidebar:
    st.title("Navigasi")
    if st.button("Halaman Utama"):
        st.session_state.halaman = "Halaman Utama"
    if st.button("Prediksi Antrikorosi"):
        st.session_state.halaman = "prediksi"
    

# Menampilkan konten sesuai dengan halaman yang aktif
if st.session_state.halaman == "Halaman Utama":
    st.title("Selamat Datang di Aplikasi Prediksi Efektivitas Inhibitor Korosi Berbasis Machine Learning Senyawa Quinoxaline")
    st.write("Aplikasi ini dirancang untuk memprediksi efektivitas senyawa inhibitor korosi menggunakan model machine learning. Mengimplementasikan model XGBoost Regressor, aplikasi ini memungkinkan pengguna untuk melakukan regresi terhadap efektivitas inhibitor berbasis quinoxaline. Dengan mampu menghasilkan nilai R2 tertinggi sebesar 0.958 dan RMSE, MSE, MAD, MAPE paling rendah dengan nilai berturut-turut 0.434, 0.189, 0.122, 0.343. Dengan nilai tersebut XGBR mampu memperlihatkan kemampuan prediksi yang baik. Tujuannya adalah memberikan prediksi tingkat proteksi yang diberikan oleh senyawa inhibitor terhadap korosi dalam berbagai kondisi.")
    
    st.image("plotxgbr.png", caption="Plot Visualisasi", use_column_width=True)
    st.write("Dapat dilihat pada Gambar diatas sebaran data poin untuk hasil prediksi sangat mendekati garis prediksi. Semakin dekat dengan garis prediksi maka model tersebut bisa menghasilkan prediksi yang baik. Hal ini menunjukkan pada kasus ini model XGBR memiliki kemampuan prediksi yang lebih unggul dengan mampu menghasilkan nilai prediksi yang mendekati nilai aktualnya.")
    st.write("Untuk menggunakan aplikasi prediksi antikorosi bisa diakses melalui menu di samping.")
elif st.session_state.halaman == "prediksi":
    st.title("Prediksi Efisiensi Senyawa Antikorosi Quinoxaline")
    # Membuat form
    with st.form("my_form"):
        st.write("Input")

        # Membuat dua kolom
        col1, col2 = st.columns(2)

        # Field input di kolom pertama
        with col1:
            ConM = st.number_input(label="Con (M)", step=0.001, format="%0.3f")
            

            TempKelvin = st.number_input(label="Temp (K)")
           

            TEeV = st.number_input(label="TE (eV)", step=0.01, format="%0.2f")
            

            EHOMOeV = st.number_input(label="EHOMO (eV)", step=0.001, format="%0.3f")
            

            ELUMOeV = st.number_input(label="ELUMO (eV)", step=0.001, format="%0.3f")
            
            deltaΕeV = st.number_input(label="ΔΕ (eV)", step=0.001, format="%0.3f")
            

            µ = st.number_input(label="µ", step=0.001, format="%0.3f")
                        

        # Field input di kolom kedua
        with col2:
            

            iP = st.number_input(label="IP", step=0.001, format="%0.3f")
            

            eA = st.number_input(label="EA", step=0.001, format="%0.3f")
           

            X = st.number_input(label="X", step=0.001, format="%0.3f")

            η = st.number_input(label="η", step=0.001, format="%0.3f")
            

            σ = st.number_input(label="σ", step=0.001, format="%0.3f")
            

            deltaΝ = st.number_input(label="ΔΝ", step=0.001, format="%0.3f")
            
        submit_button = st.form_submit_button("Submit", type="primary")
    # Tampilkan hasil input jika form disubmit
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

    prediction = 0
    if submit_button:
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

    

