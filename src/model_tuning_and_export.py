"""
Phase 3-6 — Fast Version
Strategi: Grid kecil + KFold-5 untuk tuning, KFold-10 untuk evaluasi.
Prioritas: selesai cepat dan menghasilkan model yang lebih baik dari default.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no GUI window needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json, joblib
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')
os.makedirs('outputs/figures', exist_ok=True); os.makedirs('outputs/tables', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ── Load & prep data ───────────────────────────────────────────────────
df = pd.read_csv('data/raw/impact_test_dataset_fe_a36_al_1xxx.csv')
df.columns = ['Material','T_C','a_mm','b_mm','Luas_A_mm2','E_Joule','Shear_Fracture_pct']
df['Rasio_ab']         = df['a_mm'] / df['b_mm']
df['Ketangguhan']      = df['E_Joule'] / df['Luas_A_mm2'] * 1000
df['Material_encoded'] = (df['Material'] == 'Fe A36').astype(int)
df['Kategori_Suhu']    = pd.cut(df['T_C'], bins=[-np.inf, 0, 50, np.inf],
                                 labels=['Dingin','Ruangan','Panas'])

features = ['Material_encoded','T_C','a_mm','b_mm','Luas_A_mm2','Rasio_ab']
X        = df[features].values
y_e      = df['E_Joule'].values
y_s      = df['Shear_Fracture_pct'].values

kf5  = KFold(n_splits=5, shuffle=True, random_state=42)
kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
colors_mat = {'Fe A36': '#2196F3', 'Al 1XXX': '#FF5722'}

# ── Baseline Phase 2 (SVR default, KFold-10) ──────────────────────────
svr_def = Pipeline([('sc', StandardScaler()), ('svr', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))])
y_def_e  = cross_val_predict(svr_def, X, y_e, cv=kf10)
y_def_s  = cross_val_predict(svr_def, X, y_s, cv=kf10)
r2_def_e = r2_score(y_e, y_def_e); rmse_def_e = np.sqrt(mean_squared_error(y_e, y_def_e))
r2_def_s = r2_score(y_s, y_def_s); rmse_def_s = np.sqrt(mean_squared_error(y_s, y_def_s))
print(f"Baseline SVR default — Energi: R²={r2_def_e:.4f}, RMSE={rmse_def_e:.2f}")
print(f"Baseline SVR default — Shear : R²={r2_def_s:.4f}, RMSE={rmse_def_s:.2f}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: HYPERPARAMETER TUNING — Grid kecil, KFold-5
# ══════════════════════════════════════════════════════════════════════
print("\n--- PHASE 3: TUNING ---")

# Grid KECIL tapi representatif
param_grid = {
    'svr__C':       [1, 10, 100, 500],
    'svr__gamma':   ['scale', 0.01, 0.1],
    'svr__epsilon': [0.1, 1.0, 5.0],
    'svr__kernel':  ['rbf']
}
print(f"Grid: {4*3*3*1} = 36 kombinasi × KFold-5 = 180 fits per target")

svr_pipe = Pipeline([('sc', StandardScaler()), ('svr', SVR())])

# Tuning Energi
gs_e = GridSearchCV(svr_pipe, param_grid, cv=kf5, scoring='r2', n_jobs=1)
gs_e.fit(X, y_e)
print(f"Energi best params: {gs_e.best_params_}  | CV R²={gs_e.best_score_:.4f}")

# Tuning Shear
gs_s = GridSearchCV(svr_pipe, param_grid, cv=kf5, scoring='r2', n_jobs=1)
gs_s.fit(X, y_s)
print(f"Shear  best params: {gs_s.best_params_}  | CV R²={gs_s.best_score_:.4f}")

# Evaluasi dengan KFold-10
y_tuned_e  = cross_val_predict(gs_e.best_estimator_, X, y_e, cv=kf10)
y_tuned_s  = cross_val_predict(gs_s.best_estimator_, X, y_s, cv=kf10)
r2_tuned_e  = r2_score(y_e, y_tuned_e); rmse_tuned_e = np.sqrt(mean_squared_error(y_e, y_tuned_e))
r2_tuned_s  = r2_score(y_s, y_tuned_s); rmse_tuned_s = np.sqrt(mean_squared_error(y_s, y_tuned_s))
mae_tuned_e = mean_absolute_error(y_e, y_tuned_e)
print(f"\nEvaluasi KFold-10 setelah tuning:")
print(f"  Energi: R²={r2_tuned_e:.4f}, RMSE={rmse_tuned_e:.2f}, MAE={mae_tuned_e:.2f}")
print(f"  Shear : R²={r2_tuned_s:.4f}, RMSE={rmse_tuned_s:.2f}")

# Grafik perbandingan Default vs Tuned
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('R² Score: SVR Default vs SVR Tuned (KFold-10 CV)', fontsize=15, fontweight='bold')

for ax, vals, cols, title in [
    (axes[0], [r2_def_e, r2_tuned_e], ['#90CAF9','#1565C0'], 'Target: Energi Impak (Joule)'),
    (axes[1], [r2_def_s, r2_tuned_s], ['#A5D6A7','#2E7D32'], 'Target: Shear Fracture (%)')
]:
    labels_b = ['SVR Default\n(Phase 2)', 'SVR Tuned\n(Phase 3)']
    bars = ax.bar([0,1], [max(v,-0.6) for v in vals], color=cols, width=0.45, edgecolor='white')
    ax.set_xticks([0,1]); ax.set_xticklabels(labels_b)
    ax.set_title(title); ax.set_ylabel('R² Score (KFold-10)')
    ax.axhline(0, color='red', ls='--', alpha=0.4)
    ymax = max(max(vals)+0.15, 0.7)
    ax.set_ylim(min(min(vals)-0.1,-0.15), ymax)
    for b, v in zip(bars, vals):
        offset = 0.02 if v >= 0 else -0.07
        ax.text(b.get_x()+b.get_width()/2, max(v,-0.6)+abs(offset),
                f'{v:.4f}', ha='center', fontweight='bold', fontsize=13,
                color='white' if v < -0.3 else 'black')

plt.tight_layout()
plt.savefig('outputs/figures/10_tuning_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Grafik: output/10_tuning_comparison.png")

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: FINAL MODEL TRAINING (semua data)
# ══════════════════════════════════════════════════════════════════════
print("\n--- PHASE 4: FINAL MODEL TRAINING ---")

final_e = gs_e.best_estimator_; final_e.fit(X, y_e)
final_s = gs_s.best_estimator_; final_s.fit(X, y_s)

y_tr_e = final_e.predict(X); y_tr_s = final_s.predict(X)
r2_tr_e  = r2_score(y_e, y_tr_e); rmse_tr_e = np.sqrt(mean_squared_error(y_e, y_tr_e))
r2_tr_s  = r2_score(y_s, y_tr_s); rmse_tr_s = np.sqrt(mean_squared_error(y_s, y_tr_s))
print(f"  Energi train: R²={r2_tr_e:.4f}, RMSE={rmse_tr_e:.2f} J")
print(f"  Shear  train: R²={r2_tr_s:.4f}, RMSE={rmse_tr_s:.2f} %")

# Grafik Actual vs Predicted (training)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Actual vs Predicted — Final Model SVR Tuned (Training)', fontsize=15, fontweight='bold')
for mat, col, mk in [('Fe A36','#2196F3','o'),('Al 1XXX','#FF5722','s')]:
    mask = df['Material']==mat
    axes[0].scatter(y_e[mask], y_tr_e[mask], c=col, marker=mk, s=90, alpha=0.8,
                    label=mat, edgecolors='white', lw=0.5)
    axes[1].scatter(y_s[mask], y_tr_s[mask], c=col, marker=mk, s=90, alpha=0.8,
                    label=mat, edgecolors='white', lw=0.5)
for ax, act, pred, r2, unit, title in [
    (axes[0],y_e,y_tr_e,r2_tr_e,'Joule','Energi Impak'),
    (axes[1],y_s,y_tr_s,r2_tr_s,'%','Shear Fracture')
]:
    lo=min(act.min(),pred.min()); hi=max(act.max(),pred.max())
    ax.plot([lo,hi],[lo,hi],'k--',alpha=0.5,lw=2,label='Perfect')
    ax.set_xlabel(f'Actual ({unit})'); ax.set_ylabel(f'Predicted ({unit})')
    ax.set_title(f'{title}\nR² (train) = {r2:.4f}'); ax.legend()
plt.tight_layout()
plt.savefig('outputs/figures/11_final_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Grafik: output/11_final_actual_vs_predicted.png")

# ══════════════════════════════════════════════════════════════════════
# PHASE 5: SAVING MODEL & PREDIKSI
# ══════════════════════════════════════════════════════════════════════
print("\n--- PHASE 5: SAVING MODEL ---")
joblib.dump(final_e, 'models/svr_energy_model.pkl')
joblib.dump(final_s, 'models/svr_shear_model.pkl')
params_out = {
    'energy': {k:str(v) for k,v in gs_e.best_params_.items()},
    'shear':  {k:str(v) for k,v in gs_s.best_params_.items()},
    'kfold10_r2_energy': round(r2_tuned_e,4),
    'kfold10_r2_shear':  round(r2_tuned_s,4),
    'features': features
}
with open('models/best_params.json','w') as f:
    json.dump(params_out, f, indent=2)
print("  model/svr_energy_model.pkl  [OK]")
print("  model/svr_shear_model.pkl   [OK]")
print("  model/best_params.json      [OK]")

def prediksi_impak(material, suhu_C, a_mm, b_mm):
    luas = a_mm*b_mm; rasio = a_mm/b_mm
    enc  = 1 if material=='Fe A36' else 0
    Xn   = np.array([[enc, suhu_C, a_mm, b_mm, luas, rasio]])
    e    = max(0, final_e.predict(Xn)[0])
    s    = max(0, min(100, final_s.predict(Xn)[0]))
    return {'Material':material,'Suhu(C)':suhu_C,
            'Prediksi_E(J)':round(e,2),'Prediksi_Shear(%)':round(s,2),
            'Ketangguhan(kJ/m2)':round(e/luas*1000,2)}

demo_cases = [
    ('Fe A36',-30,9,10),('Fe A36',25,9,10),('Fe A36',150,9,10),
    ('Al 1XXX',-30,9,10),('Al 1XXX',25,9,10),('Al 1XXX',150,9,10),
]
demo_results = [prediksi_impak(*c) for c in demo_cases]
df_demo = pd.DataFrame(demo_results)
print("\nDEMO PREDIKSI:")
print(df_demo.to_string(index=False))

# Kurva prediksi
fig, axes = plt.subplots(1,2,figsize=(14,6))
fig.suptitle('Kurva Prediksi SVR Tuned — Suhu vs Performa Impak', fontsize=15, fontweight='bold')
T_line = np.linspace(-40,220,300)
a_r,b_r = 9.0,10.0; lr=a_r*b_r; rr=a_r/b_r
for mat, enc in [('Fe A36',1),('Al 1XXX',0)]:
    Xc = np.array([[enc,t,a_r,b_r,lr,rr] for t in T_line])
    ec = np.clip(final_e.predict(Xc),0,None)
    sc = np.clip(final_s.predict(Xc),0,100)
    axes[0].plot(T_line,ec,color=colors_mat[mat],lw=2.5,label=f'{mat} (prediksi)')
    axes[1].plot(T_line,sc,color=colors_mat[mat],lw=2.5,label=f'{mat} (prediksi)')
    sub = df[df['Material']==mat]
    axes[0].scatter(sub['T_C'],sub['E_Joule'],color=colors_mat[mat],s=55,alpha=0.45,edgecolors='white',zorder=5)
    axes[1].scatter(sub['T_C'],sub['Shear_Fracture_pct'],color=colors_mat[mat],s=55,alpha=0.45,edgecolors='white',zorder=5)
for ax, yl, title in [(axes[0],'Energi Impak (Joule)','Suhu vs Energi Impak'),
                       (axes[1],'Shear Fracture (%)','Suhu vs Shear Fracture')]:
    ax.set_xlabel('Suhu (C)'); ax.set_ylabel(yl); ax.set_title(title)
    ax.legend(); ax.axvline(0,color='gray',ls=':',alpha=0.5)
plt.tight_layout()
plt.savefig('outputs/figures/12_prediction_curve.png',dpi=150,bbox_inches='tight')
plt.close()
print("Grafik: output/12_prediction_curve.png")

# ══════════════════════════════════════════════════════════════════════
# PHASE 6: EXPORT CSV UNTUK POWER BI
# ══════════════════════════════════════════════════════════════════════
print("\n--- PHASE 6: EXPORT CSV ---")

# 1. hasil prediksi final (prediksi dari KFold-10 CV — lebih jujur)
df_out = df.copy()
df_out['Prediksi_E_Final']     = np.clip(y_tuned_e, 0, None).round(2)
df_out['Prediksi_Shear_Final'] = np.clip(y_tuned_s, 0, 100).round(2)
df_out['Residual_E']           = (df_out['E_Joule'] - df_out['Prediksi_E_Final']).round(2)
df_out['Residual_Shear']       = (df_out['Shear_Fracture_pct'] - df_out['Prediksi_Shear_Final']).round(2)
df_out['AbsError_pct_E']       = (abs(df_out['Residual_E'])/df_out['E_Joule']*100).round(2)
df_out.to_csv('outputs/tables/final_prediction_results.csv', index=False)
print("  output/final_prediction_results.csv  [OK]")

# 2. Model summary
pd.DataFrame({
    'Tahap':        ['Phase 2 — Default','Phase 3 — Tuned','Phase 4 — Full Train'],
    'Model':        ['SVR RBF default','SVR RBF tuned','SVR RBF tuned'],
    'Validasi':     ['KFold-10','KFold-10','Training Score'],
    'R2_Energi':    [round(r2_def_e,4), round(r2_tuned_e,4), round(r2_tr_e,4)],
    'RMSE_Energi':  [round(rmse_def_e,2),round(rmse_tuned_e,2),round(rmse_tr_e,2)],
    'MAE_Energi':   [round(mean_absolute_error(y_e,y_def_e),2),
                     round(mae_tuned_e,2),
                     round(mean_absolute_error(y_e,y_tr_e),2)],
    'R2_Shear':     [round(r2_def_s,4), round(r2_tuned_s,4), round(r2_tr_s,4)],
    'RMSE_Shear':   [round(rmse_def_s,2),round(rmse_tuned_s,2),round(rmse_tr_s,2)],
}).to_csv('outputs/tables/model_summary.csv', index=False)
print("  output/model_summary.csv         [OK]")

# 3. Demo prediksi
df_demo.to_csv('outputs/tables/demo_predictions.csv', index=False)
print("  output/demo_predictions.csv          [OK]")

# 4. Feature importance
rf_e = RandomForestRegressor(n_estimators=200, random_state=42); rf_e.fit(X, y_e)
rf_s = RandomForestRegressor(n_estimators=200, random_state=42); rf_s.fit(X, y_s)
feat_labels = {'Material_encoded':'Jenis Material','T_C':'Suhu (C)',
               'a_mm':'Lebar Takik a (mm)','b_mm':'Tinggi b (mm)',
               'Luas_A_mm2':'Luas Penampang (mm2)','Rasio_ab':'Rasio a/b'}
pd.DataFrame({'Feature':features,
              'Feature_Label':[feat_labels[f] for f in features],
              'Importance_Energy':rf_e.feature_importances_.round(4),
              'Importance_Shear': rf_s.feature_importances_.round(4)
}).to_csv('outputs/tables/feature_importance_final.csv', index=False)
print("  output/feature_importance_final.csv  [OK]")

print(f"""
=================================================================
  RINGKASAN HASIL AKHIR
=================================================================
  TARGET 1 — Energi Impak (Joule):
    Model     : SVR RBF (Tuned, {gs_e.best_params_})
    KFold-10 R²  : {r2_tuned_e:.4f}
    KFold-10 RMSE: {rmse_tuned_e:.2f} Joule
    KFold-10 MAE : {mae_tuned_e:.2f} Joule

  TARGET 2 — Shear Fracture (%):
    Model     : SVR RBF (Tuned, {gs_s.best_params_})
    KFold-10 R²  : {r2_tuned_s:.4f}
    KFold-10 RMSE: {rmse_tuned_s:.2f} %

  SEMUA OUTPUT TERSIMPAN DI:
    output/final_prediction_results.csv
    output/model_summary.csv
    output/demo_predictions.csv
    output/feature_importance_final.csv
    output/10_tuning_comparison.png
    output/11_final_actual_vs_predicted.png
    output/12_prediction_curve.png
    model/svr_energy_model.pkl
    model/svr_shear_model.pkl
=================================================================
""")
