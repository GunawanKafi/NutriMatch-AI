import streamlit as st
import pandas as pd
import joblib

# Pengaturan Halaman
st.set_page_config(page_title="Rekomendasi Gizi ML", page_icon="🍏", layout="centered")

# --- PEMILIHAN BAHASA DARI SIDEBAR ---
bahasa = st.sidebar.radio("🌐 Language / Bahasa", ["Indonesia", "English"])

# --- KAMUS TEKS DINAMIS ---
if bahasa == "Indonesia":
    teks = {
        "judul": "🍏 Sistem Klasifikasi Gizi Machine Learning",
        "penafian": "**Penafian:** Aplikasi prototipe Machine Learning untuk tugas akademik. Tidak ditujukan sebagai rujukan medis klinis.",
        "subheader_input": "Masukkan Profil & Kebutuhan Anda",
        "pilih_kategori": "Pilih Kategori",
        "pilih_darah": "Pilih Golongan Darah",
        "fokus_pencegahan": "Fokus Pencegahan",
        "tombol_cari": "Cari Rekomendasi ML",
        "error_kosong": "Maaf, tidak ada data {jenis} untuk pencegahan {penyakit} di dalam database.",
        "subheader_hasil": "Hasil Analisis Algoritma KNN:",
        "sukses_ditemukan": "Ditemukan **{jumlah} {jenis}** yang direkomendasikan dan cocok dengan Anda:",
        "peringatan_blokir": "Ada item untuk penyakit tersebut, namun algoritma memprediksi item tersebut berstatus '0 (Netral)' atau '-1 (Pantangan)' untuk golongan darah Anda.",
        "lang_code": "Indonesia"
    }
else:
    teks = {
        "judul": "🍏 ML Nutritional Classification System",
        "penafian": "**Disclaimer:** Machine Learning prototype application for academic purposes. Not intended as a clinical medical reference.",
        "subheader_input": "Enter Your Profile & Needs",
        "pilih_kategori": "Select Category",
        "pilih_darah": "Select Blood Type",
        "fokus_pencegahan": "Prevention Focus",
        "tombol_cari": "Search ML Recommendations",
        "error_kosong": "Sorry, no {jenis} data found for preventing {penyakit} in the database.",
        "subheader_hasil": "KNN Algorithm Analysis Results:",
        "sukses_ditemukan": "Found **{jumlah} recommended {jenis}** that suit you:",
        "peringatan_blokir": "Items found for this disease, but the algorithm predicts they are '0 (Neutral)' or '-1 (Avoid)' for your blood type.",
        "lang_code": "English"
    }

st.title(teks["judul"], anchor=False)
st.warning(teks["penafian"])
st.markdown("---")

@st.cache_resource
def load_resources():
    df = pd.read_csv('models/dataset_clean.csv')
    knn_model = joblib.load('models/knn_model.pkl')
    le_jenis = joblib.load('models/le_jenis.pkl')
    le_darah = joblib.load('models/le_darah.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return df, knn_model, le_jenis, le_darah, scaler

df_full, knn_model, le_jenis, le_darah, scaler = load_resources()

# --- FILTER DATA BERDASARKAN BAHASA ---
df_aktif = df_full[df_full['Bahasa'] == teks["lang_code"]]

# Ekstraksi Penyakit (Opsi B)
semua_penyakit = df_aktif['Target_Pencegahan'].str.split(',').explode().str.strip().unique()
semua_penyakit = sorted(semua_penyakit) 

st.subheader(teks["subheader_input"], anchor=False)

col1, col2 = st.columns(2)
with col1:
    jenis_input = st.selectbox(teks["pilih_kategori"], df_aktif['Jenis'].unique())
    darah_input = st.selectbox(teks["pilih_darah"], ['A', 'B', 'O', 'AB'])
with col2:
    penyakit_input = st.selectbox(teks["fokus_pencegahan"], semua_penyakit)

st.markdown("---")

if st.button(teks["tombol_cari"], type="primary"):
    
    # Logika Filter
    df_filtered = df_aktif[
        (df_aktif['Jenis'] == jenis_input) & 
        (df_aktif['Target_Pencegahan'].str.contains(penyakit_input, case=False, na=False))
    ]
    
    if df_filtered.empty:
        st.error(teks["error_kosong"].format(jenis=jenis_input.lower(), penyakit=penyakit_input))
    else:
        hasil_rekomendasi = []
        
        # --- PENYELARASAN ML ---
        # Memastikan mesin ML membaca bahasa Inggris (Fruit) sebagai bahasa mesin (Buah)
        pemetaan_jenis = {'Buah': 'Buah', 'Sayur': 'Sayur', 'Fruit': 'Buah', 'Vegetable': 'Sayur'}
        jenis_standar = pemetaan_jenis.get(jenis_input, 'Buah')
        
        for index, row in df_filtered.iterrows():
            gizi_df = pd.DataFrame([[row['Serat_g'], row['Vitamin_C_mg'], row['Vitamin_A_IU']]], 
                                    columns=['Serat_g', 'Vitamin_C_mg', 'Vitamin_A_IU'])
            gizi_scaled = scaler.transform(gizi_df)
            
            # Encoding kategori menggunakan jenis yang sudah distandarkan
            j_enc = le_jenis.transform([jenis_standar])[0]
            d_enc = le_darah.transform([darah_input])[0]
            
            fitur_akhir = [[gizi_scaled[0][0], gizi_scaled[0][1], gizi_scaled[0][2], j_enc, d_enc]]
            
            prediksi = knn_model.predict(fitur_akhir)[0]
            
            if prediksi == 1:
                hasil_rekomendasi.append(row['Nama_Pangan'])
        
        st.subheader(teks["subheader_hasil"], anchor=False)
        if len(hasil_rekomendasi) > 0:
            st.success(teks["sukses_ditemukan"].format(jumlah=len(hasil_rekomendasi), jenis=jenis_input.lower()))
            
            # --- LOGIKA PEMBUATAN TABEL DAN TRANSLASI KONDISI ---
            data_tabel = []
            
            # Kamus terjemahan untuk metode pengolahan
            kamus_kondisi_id = {
                'raw': 'Mentah',
                'cooked': 'Dimasak',
                'baked': 'Dipanggang',
                'stewed': 'Direbus',
                'dry-roasted': 'Disangrai'
            }
            
            for res in hasil_rekomendasi:
                # Cek apakah ada tanda koma di nama pangan
                if ',' in res:
                    parts = res.split(',')
                    nama_item = parts[0].strip()
                    kondisi_asli = parts[1].strip().lower()
                    
                    # Terjemahkan jika bahasa Indonesia
                    if bahasa == "Indonesia":
                        kondisi_tampil = kamus_kondisi_id.get(kondisi_asli, kondisi_asli.capitalize())
                    else:
                        kondisi_tampil = kondisi_asli.capitalize()
                else:
                    # Jika tidak ada koma, asumsikan bentuk umum/mentah
                    nama_item = res
                    if bahasa == "Indonesia":
                        kondisi_tampil = "Umum / Mentah"
                    else:
                        kondisi_tampil = "General / Raw"
                        
                # Menambahkan data ke baris tabel dengan header sesuai bahasa
                if bahasa == "Indonesia":
                    data_tabel.append({"Nama Pangan": nama_item, "Metode Konsumsi": kondisi_tampil})
                else:
                    data_tabel.append({"Food Name": nama_item, "Consumption Method": kondisi_tampil})
                    
            # Mengubah list menjadi DataFrame dan menampilkannya sebagai tabel statis
            df_hasil = pd.DataFrame(data_tabel)
            
            # Menghilangkan indeks baris (angka 0, 1, 2 di sebelah kiri) agar lebih bersih
            st.table(df_hasil.assign(Index='').set_index('Index'))
            st.markdown("---")
            
            # 1. Menentukan judul grafik berdasarkan bahasa
            if bahasa == "Indonesia":
                st.subheader("📊 Perbandingan Gizi Makanan yang Direkomendasikan", anchor=False)
            else:
                st.subheader("📊 Nutritional Comparison of Recommended Foods", anchor=False)
                
            # 2. Mengambil data nutrisi mentah dari dataframe asli khusus untuk makanan yang direkomendasikan
            df_visual = df_aktif[df_aktif['Nama_Pangan'].isin(hasil_rekomendasi)].copy()
            
            # 3. Menyiapkan label kolom berdasarkan bahasa agar rapi di grafik
            if bahasa == "Indonesia":
                df_visual = df_visual.rename(columns={
                    'Nama_Pangan': 'Nama Pangan',
                    'Serat_g': 'Serat (g)', 
                    'Vitamin_C_mg': 'Vitamin C (mg)', 
                    'Vitamin_A_IU': 'Vitamin A (IU)'
                })
                kolom_nutrisi = ['Serat (g)', 'Vitamin C (mg)', 'Vitamin A (IU)']
                kolom_index = 'Nama Pangan'
            else:
                df_visual = df_visual.rename(columns={
                    'Nama_Pangan': 'Food Name',
                    'Serat_g': 'Fiber (g)', 
                    'Vitamin_C_mg': 'Vitamin C (mg)', 
                    'Vitamin_A_IU': 'Vitamin A (IU)'
                })
                kolom_nutrisi = ['Fiber (g)', 'Vitamin C (mg)', 'Vitamin A (IU)']
                kolom_index = 'Food Name'
            
            # 4. Menyaring hanya kolom yang dibutuhkan dan mengatur index untuk grafik
            df_visual = df_visual[[kolom_index] + kolom_nutrisi]
            df_visual.set_index(kolom_index, inplace=True)
            
            # 5. Merender grafik batang secara terpisah menggunakan Tabs
            if bahasa == "Indonesia":
                tab1, tab2, tab3 = st.tabs(["Serat (g)", "Vitamin C (mg)", "Vitamin A (IU)"])
                with tab1:
                    st.bar_chart(df_visual[['Serat (g)']])
                with tab2:
                    st.bar_chart(df_visual[['Vitamin C (mg)']])
                with tab3:
                    st.bar_chart(df_visual[['Vitamin A (IU)']])
            else:
                tab1, tab2, tab3 = st.tabs(["Fiber (g)", "Vitamin C (mg)", "Vitamin A (IU)"])
                with tab1:
                    st.bar_chart(df_visual[['Fiber (g)']])
                with tab2:
                    st.bar_chart(df_visual[['Vitamin C (mg)']])
                with tab3:
                    st.bar_chart(df_visual[['Vitamin A (IU)']])
        else:
            st.warning(teks["peringatan_blokir"])