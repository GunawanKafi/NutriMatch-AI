import pandas as pd

# 1. Baca kedua dataset (Sesuaikan nama filenya dengan milikmu)
df_id = pd.read_csv('data_gizi_golongan_darah_lengkap_IDN.csv', sep=';')
df_en = pd.read_csv('data_gizi_golongan_darah_lengkap_ENG.csv', sep=';')

# # 2. Tambahkan kolom penanda bahasa
# df_id['Bahasa'] = 'ID'
# df_en['Bahasa'] = 'EN'

# 3. Tumpuk menjadi 1 dataset secara vertikal
df_gabungan = pd.concat([df_id, df_en], ignore_index=True)

# 4. Simpan menjadi dataset bilingual baru
df_gabungan.to_csv('data_gizi_bilingual.csv', sep=';', index=False)
print("Dataset bilingual berhasil dibuat!")