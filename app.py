import streamlit as st
import pandas as pd
import joblib

# Pengaturan Halaman
st.set_page_config(page_title="Rekomendasi Gizi ML", page_icon="🍏", layout="centered")
st.title("🍏 Sistem Klasifikasi Gizi Machine Learning")
st.warning("**Penafian:** Aplikasi prototipe Machine Learning untuk tugas akademik. Tidak ditujukan sebagai rujukan medis klinis.")
st.markdown("---")

@st.cache_resource
def load_resources():
    # Memuat dataset murni dan model yang sudah disiapkan di Fase 2 & 3
    df = pd.read_csv('models/dataset_clean.csv')
    knn_model = joblib.load('models/knn_model.pkl')
    le_jenis = joblib.load('models/le_jenis.pkl')
    le_darah = joblib.load('models/le_darah.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return df, knn_model, le_jenis, le_darah, scaler

df, knn_model, le_jenis, le_darah, scaler = load_resources()

# --- EKSTRAKSI PENYAKIT (OPSI B) ---
# Memisahkan teks berdasarkan koma, membersihkan spasi, dan mengambil nilai uniknya
semua_penyakit = df['Target_Pencegahan'].str.split(',').explode().str.strip().unique()
semua_penyakit = sorted(semua_penyakit) # Mengurutkan secara alfabet agar rapi di dropdown

st.subheader("Masukkan Profil & Kebutuhan Anda")

col1, col2 = st.columns(2)
with col1:
    jenis_input = st.selectbox("Pilih Kategori", df['Jenis'].unique())
    darah_input = st.selectbox("Pilih Golongan Darah", ['A', 'B', 'O', 'AB'])
with col2:
    penyakit_input = st.selectbox("Fokus Pencegahan", semua_penyakit)

st.markdown("---")

if st.button("Cari Rekomendasi ML", type="primary"):
    
    # --- LOGIKA PENCARIAN OPSI B ---
    # Memfilter data dimana Jenis sesuai, DAN Target Pencegahan 'mengandung' kata yang dipilih
    df_filtered = df[
        (df['Jenis'] == jenis_input) & 
        (df['Target_Pencegahan'].str.contains(penyakit_input, case=False, na=False))
    ]
    
    if df_filtered.empty:
        st.error(f"Maaf, tidak ada data {jenis_input.lower()} untuk pencegahan {penyakit_input} di dalam database.")
    else:
        hasil_rekomendasi = []
        
        # Looping untuk memprediksi skor setiap item yang lolos filter pencarian
        for index, row in df_filtered.iterrows():
            # Mengambil dan menormalkan data gizi (Scaling)
            gizi_df = pd.DataFrame([[row['Serat_g'], row['Vitamin_C_mg'], row['Vitamin_A_IU']]], 
                                     columns=['Serat_g', 'Vitamin_C_mg', 'Vitamin_A_IU'])
            gizi_scaled = scaler.transform(gizi_df)
            
            # Encoding kategori
            j_enc = le_jenis.transform([row['Jenis']])[0]
            d_enc = le_darah.transform([darah_input])[0]
            
            # Menyusun format fitur: [Serat_g, Vit_C, Vit_A, Jenis, Darah]
            fitur_akhir = [[gizi_scaled[0][0], gizi_scaled[0][1], gizi_scaled[0][2], j_enc, d_enc]]
            
            # Melakukan Prediksi
            prediksi = knn_model.predict(fitur_akhir)[0]
            
            # Jika prediksi bernilai 1 (Sangat Baik / Bermanfaat)
            if prediksi == 1:
                hasil_rekomendasi.append(row['Nama_Pangan'])
        
        # --- MENAMPILKAN HASIL ---
        st.subheader("Hasil Analisis Algoritma KNN:")
        if len(hasil_rekomendasi) > 0:
            st.success(f"Ditemukan **{len(hasil_rekomendasi)} {jenis_input.lower()}** yang direkomendasikan dan cocok dengan Anda:")
            for res in hasil_rekomendasi:
                st.markdown(f"- {res}")
        else:
            st.warning("Ada item untuk penyakit tersebut, namun algoritma memprediksi item tersebut berstatus '0 (Netral)' atau '-1 (Pantangan)' untuk golongan darah Anda.")