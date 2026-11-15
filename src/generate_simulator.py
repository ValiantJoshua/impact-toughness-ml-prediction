"""
generate_simulator.py
=====================
Menghasilkan file 'simulator_predictions.csv' berisi prediksi SVR
untuk berbagai variasi SUHU, dengan dimensi spesimen tetap (standar).

Dimensi standar (median dari data asli):
  a_mm       = 8.76  mm
  b_mm       = 10.00 mm
  Luas_A_mm2 = 87.44 mm²
  Rasio_ab   = 0.876

Output:
  output/simulator_predictions.csv  — siap diimpor ke Power BI
"""

import numpy as np
import pandas as pd
import joblib
import os

# ── Muat model SVR yang sudah tersimpan ──────────────────────────────────────
print("Memuat model SVR dari folder 'models/'...")
model_energi = joblib.load('models/svr_energy_model.pkl')
model_shear  = joblib.load('models/svr_shear_model.pkl')
print("  [OK] svr_energy_model.pkl")
print("  [OK] svr_shear_model.pkl")

# ── Dimensi standar (tetap, sesuai standar pengujian Charpy) ─────────────────
a_mm       = 8.76    # lebar takik (mm) — median dari data asli
b_mm       = 10.00   # tinggi spesimen (mm)
Luas_A     = a_mm * b_mm   # = 87.60 mm²
Rasio_ab   = a_mm / b_mm   # = 0.876

print(f"\nDimensi standar yang digunakan:")
print(f"  a_mm       = {a_mm} mm")
print(f"  b_mm       = {b_mm} mm")
print(f"  Luas_A_mm2 = {Luas_A:.2f} mm²")
print(f"  Rasio_ab   = {Rasio_ab:.4f}")

# ── Rentang suhu simulasi ─────────────────────────────────────────────────────
# -60°C s/d 300°C dengan step 1°C = 361 titik per material = 722 baris total
suhu_range = np.arange(-60, 301, 1)   # -60, -59, ..., 300
materials  = ['Fe A36', 'Al 1XXX']

print(f"\nRentang suhu: {suhu_range[0]}°C s/d {suhu_range[-1]}°C (step 1°C)")
print(f"Total baris  : {len(suhu_range) * len(materials)} baris")

# ── Buat semua kombinasi ──────────────────────────────────────────────────────
rows = []
for mat in materials:
    mat_encoded = 1 if mat == 'Fe A36' else 0
    for t in suhu_range:
        rows.append({
            'Material':         mat,
            'Material_encoded': mat_encoded,
            'T_C':              t,
            'a_mm':             a_mm,
            'b_mm':             b_mm,
            'Luas_A_mm2':       Luas_A,
            'Rasio_ab':         Rasio_ab,
        })

df_sim = pd.DataFrame(rows)

# ── Urutan fitur HARUS sama persis dengan saat model dilatih ─────────────────
features = ['Material_encoded', 'T_C', 'a_mm', 'b_mm', 'Luas_A_mm2', 'Rasio_ab']
X_sim = df_sim[features].values

# ── Jalankan prediksi ─────────────────────────────────────────────────────────
print("\nMenjalankan prediksi...")
df_sim['Prediksi_E_Joule']   = model_energi.predict(X_sim).round(2)
df_sim['Prediksi_Shear_pct'] = model_shear.predict(X_sim).round(2)

# Clamp nilai negatif ke 0 (secara fisik energi & shear tidak bisa negatif)
df_sim['Prediksi_E_Joule']   = df_sim['Prediksi_E_Joule'].clip(lower=0)
df_sim['Prediksi_Shear_pct'] = df_sim['Prediksi_Shear_pct'].clip(lower=0, upper=100)

# ── Hitung Ketangguhan (Energi / Luas Penampang) ──────────────────────────────
df_sim['Prediksi_Ketangguhan_kJ_m2'] = (df_sim['Prediksi_E_Joule'] / Luas_A * 1000).round(2)

# ── Tambah label Kategori Suhu (untuk filter tambahan di Power BI) ────────────
def kategori_suhu(t):
    if t < 0:
        return 'Dingin (< 0°C)'
    elif t <= 50:
        return 'Ruangan (0-50°C)'
    else:
        return 'Panas (> 50°C)'

df_sim['Kategori_Suhu'] = df_sim['T_C'].apply(kategori_suhu)

# ── Pilih kolom yang relevan untuk Power BI ───────────────────────────────────
output_cols = [
    'Material',
    'T_C',
    'Kategori_Suhu',
    'Prediksi_E_Joule',
    'Prediksi_Shear_pct',
    'Prediksi_Ketangguhan_kJ_m2',
]

df_output = df_sim[output_cols]

# ── Simpan ke CSV ─────────────────────────────────────────────────────────────
os.makedirs('outputs/figures', exist_ok=True); os.makedirs('outputs/tables', exist_ok=True)
output_path = 'outputs/tables/simulator_predictions.csv'
df_output.to_csv(output_path, index=False)

print(f"  [OK] Prediksi selesai.")
print(f"\n{'='*55}")
print(f"File disimpan: {output_path}")
print(f"Total baris  : {len(df_output)}")
print(f"{'='*55}")
print("\nContoh hasil prediksi:")
print(df_output.groupby('Material').head(3).to_string(index=False))
print("\nScript selesai. File siap diimpor ke Power BI.")
