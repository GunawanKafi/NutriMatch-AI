import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

print("--- Memulai Fase 2: Pra-pemrosesan Data ---")

# 1. Memuat Dataset Baru (Perhatikan pemisahnya sekarang menggunakan titik koma ';')
df = pd.read_csv('data_gizi_bilingual.csv', sep=';')
df.columns = df.columns.str.strip()

# 2. Teknik Melting
df_melted = pd.melt(df, 
                    id_vars=['Nama_Pangan', 'Serat_g', 'Vitamin_C_mg', 'Vitamin_A_IU', 'Jenis', 'Target_Pencegahan'],
                    value_vars=['Skor_A', 'Skor_B', 'Skor_O', 'Skor_AB'],
                    var_name='Golongan_Darah', 
                    value_name='Skor')

df_melted['Golongan_Darah'] = df_melted['Golongan_Darah'].str.replace('Skor_', '')

# --- TAMBAHAN PENTING UNTUK BILINGUAL ---
# Menstandarkan teks agar model ML menyatukan konsep 'Fruit' dan 'Buah'
pemetaan_jenis = {
    'Buah': 'Buah', 'Sayur': 'Sayur', 
    'Fruit': 'Buah', 'Vegetable': 'Sayur'
}
df_melted['Jenis_Standard'] = df_melted['Jenis'].map(pemetaan_jenis)

# 3. Proses Encoding untuk teks
le_jenis = LabelEncoder()
le_darah = LabelEncoder()

# PENTING: Latih encoder menggunakan Jenis_Standard, bukan Jenis asli
df_melted['Jenis_Encoded'] = le_jenis.fit_transform(df_melted['Jenis_Standard'])
df_melted['Golongan_Darah_Encoded'] = le_darah.fit_transform(df_melted['Golongan_Darah'])

# 4. Standard Scaler untuk data numerik (Gizi)
scaler = StandardScaler()
fitur_numerik = ['Serat_g', 'Vitamin_C_mg', 'Vitamin_A_IU']
df_melted[fitur_numerik] = scaler.fit_transform(df_melted[fitur_numerik])

# Menyimpan semua encoder dan scaler
joblib.dump(le_jenis, 'models/le_jenis.pkl')
joblib.dump(le_darah, 'models/le_darah.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Fase 2 Selesai!\n")

print("--- Memulai Fase 3: Pelatihan Model (KNN) ---")

# Menentukan X (Fitur Gizi & Teks Encoded) dan y (Skor 1, 0, atau -1)
X = df_melted[['Serat_g', 'Vitamin_C_mg', 'Vitamin_A_IU', 'Jenis_Encoded', 'Golongan_Darah_Encoded']]
y = df_melted['Skor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih KNN (Kita menggunakan n_neighbors=5 karena data sekarang lebih banyak)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

akurasi = accuracy_score(y_test, knn_model.predict(X_test))
print(f"Akurasi Model KNN Dataset Baru: {akurasi * 100:.2f}%")

joblib.dump(knn_model, 'models/knn_model.pkl')
# Menyimpan data murni ke file baru agar Streamlit lebih mudah membacanya
df.to_csv('models/dataset_clean.csv', index=False) 

print("Fase 3 Selesai! Model KNN berhasil disimpan.")