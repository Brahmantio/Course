import streamlit as st
import pandas as pd
import pickle
import time
from custom_model import CustomModelClass
import joblib
from PIL import Image

st.set_page_config(page_title="House Price Prediction", layout="centered", initial_sidebar_state="auto", page_icon="🏠")
st.title("""
Welcome to my portofolio Data Analyst

HOUSE PRICE PREDICITION
\ndashboard was created by [Bramantio](https://www.linkedin.com/in/brahmantio-w/), here I want to try to introduce the results of my portfolio or my abilities in the field of data science. This platform aims to provide an introduction, utilization, and exploration resources in the world of machine learning
""")
img = Image.open("rumah1.JPG")
st.image(img, width=500)
add_selectitem = st.sidebar.header("Prediction with CSV file")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        
    # Membuat tab untuk aplikasi
tab1, tab2, tab3 = st.tabs(["Start Prediction", "About Prediction", "How to Use"])
with tab1:
    st.header("Input your specific data")
    daerah = st.text_input("Which area do you live in?")
    crim= st.slider("Tingkat Kejahatan",
                    min_value=0,
                    max_value=88,
                    step=1,
                    value=44)
    zn= st.slider("Proporsi lahan pemukiman",
                    min_value=0,
                    max_value=100,
                    step=1,
                    value=60)
    indus= st.slider("Proporsi lahan pemukiman diatas 25.000 kaki",
                    min_value=0.46,
                    max_value=30.00,
                    step=0.1,
                    value=0.46)
    chas= st.radio("Apakah pemungkiman dekat dengan sungai?",
                    ("No", "Yes"))
    nox= st.slider("Konsentrasi oksida nitrogen (jumlah NO dan NO2)",
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    value=0.2)
    rm= st.slider("Rata-rata jumlah kamar per-rumah",
                    min_value=2,
                    max_value=8,
                    step=1,
                    value=2)
    age= st.slider("Usia rumah",
                    min_value=2,
                    max_value=100,
                    step=1,
                    value=1)
    dis= st.slider("Jarak ke pusat perkantoran",
                    min_value=2,
                    max_value=100,
                    step=1,
                    value=2)
    tax= st.slider("Tarif pajak properti",
                    min_value=100,
                    max_value=1000,
                    step=1,
                    value=0)
    ptratio= st.slider("Rasio murid-guru per kota",
                    min_value=10.1,
                    max_value=22.0,
                    step=1.1,
                    value=10.1)
    b= st.slider("proporsi penduduk warga negara asing perkota",
                    min_value=0.32,
                    max_value=396.9,
                    step=0.32,
                    value=50.00)
    lstat= st.slider("Persentase status rendah dari populasi",
                    min_value=1.73,
                    max_value=37.97,
                    step=1.00,
                    value=10.00)
    
    data = {'CRIM': crim,
            'ZN': zn,
            'INDUS': indus,
            'CHAS': 1 if chas == "Yes" else 0,
            'NOX': nox,
            'RM': rm,
            'AGE': age,
            'DIS': dis,
            'TAX': tax,
            'PTRATIO': ptratio,
            'B': b,
            'LSTAT': lstat}
    features = pd.DataFrame(data, index=[0])
    
    # Predict Button
    if st.button('Predict Now!'):
        #model_path = "/Users/bramantiow/Documents/Bootcamp/DQLAB MACHINE LEARNING/SESI 15/modeldqlab.pkl"
        #with open("modeldqlab", 'rb') as file:
            #loaded_model = pickle.load(file)
            
        model_path = "/Users/bramantiow/Documents/Bootcamp/DQLAB MACHINE LEARNING/SESI 15/modeldqlab.pkl"
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
        
        # Predicting the house price
        prediction = loaded_model.predict(features)
        
        
        # Displaying the result
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Hasil prediksiku: harga rumah di {daerah} seharga ${prediction[0]:,.2f}")

        
            
    with tab2:
        st.header("Category explanation")
        st.write("CRIM - Tingkat kejahatan per kapita per kota")
        st.write("ZN - Proporsi lahan pemukiman yang dizonasi untuk lot berukuran di atas 25.000 kaki persegi")
        st.write("INDUS - Proporsi lahan bisnis non-ritel per kota")
        st.write("CHAS - Variabel dummy sungai Charles (1 jika daerah berbatasan dengan sungai, 0 jika tidak)")
        st.write("NOX - Konsentrasi oksida nitrogen (bagian per 10 juta)")
        st.write("RM - Rata-rata jumlah kamar per rumah") 
        st.write("AGE - Proporsi unit yang dihuni pemilik yang dibangun sebelum tahun 1940")
        st.write("DIS - Jarak tertimbang ke lima pusat pekerjaan Boston")
        st.write("RAD - Indeks aksesibilitas ke jalan raya radial")
        st.write("TAX - Tarif pajak properti bernilai penuh per $10.000")
        st.write("PTRATIO - Rasio murid-guru per kota")
        st.write("B - 1000(Bk - 0.63)^2 di mana Bk adalah proporsi penduduk kulit hitam per kota")
        st.write("LSTAT - Persentase status rendah dari populasi")
        st.write("MEDV - Nilai tengah rumah yang dihuni pemilik dalam ribuan dolar")
    with tab3:
        st.header("How to use this application")
        st.write("1. Apabila ingin  memprediksi menggunakan file, pastikan file tersebut dalam format .csv dan seluruh atribut sama")
        st.write("2. Supaya prediksi akurat, pastikan nilai yang diinput sudah benar atau sesuai dengan perhitungan")
        st.write("3. Apabila sudah terisi sesuai dengan atribut, tekan tombol 'Predict Now!' untuk memulai")
        st.write("4. Hasil output berupa keterangan nominal harga dalam satuan dollar")
