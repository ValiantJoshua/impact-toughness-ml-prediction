# Impact Toughness Prediction for Fe A36 & Al 1XXX Using Machine Learning

An end-to-end Machine Learning and Business Intelligence project to predict **Charpy Impact Energy (`E_Joule`)** and **Shear Fracture Percentage (`Shear_Fracture_pct`)** for **Fe A36** structural steel and **Al 1XXX** aluminum alloy across varying test temperatures, paired with an interactive **Power BI** simulator dashboard.

## Project Structure

```text
impact-toughness-ml-prediction/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       └── impact_test_dataset_fe_a36_al_1xxx.csv
├── notebooks/
│   └── Analysis.ipynb
├── src/
│   ├── analysis.py
│   ├── analysis_full.py
│   ├── model_tuning_and_export.py
│   └── generate_simulator.py
├── models/
│   ├── best_params.json
│   ├── svr_energy_model.pkl
│   └── svr_shear_model.pkl
├── outputs/
│   ├── figures/          # EDA, model comparison, residual, and prediction plots
│   └── tables/           # Evaluation summaries and simulator prediction CSVs
└── dashboard/
    └── Power BI Dashboard.pbix
```

## Workflow Summary

1. **Data Collection & Preparation:** 54 Charpy impact test specimens (27 Fe A36, 27 Al 1XXX) with engineered features including impact toughness ($E / A$), dimension ratio ($a / b$), temperature categories, binary material encoding, and `StandardScaler` inside a scikit-learn `Pipeline`.
2. **Model Benchmarking (LOOCV):** Compared Linear Regression, Ridge, Polynomial Regression, Random Forest, Gradient Boosting, and Support Vector Regression (SVR). **SVR (RBF kernel)** achieved the best performance.
3. **Hyperparameter Tuning & Export:** Tuned SVR hyperparameters using `GridSearchCV` and exported trained pipelines to `models/svr_energy_model.pkl` and `models/svr_shear_model.pkl`.
4. **Prediction Simulator & Power BI:** Generated 722 temperature simulation rows (`-60°C` to `300°C`, step `1°C`) into `outputs/tables/simulator_predictions.csv` for interactive exploration in `dashboard/Power BI Dashboard.pbix`.

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Scripts (from the repository root)
```bash
# Run full EDA, model benchmarking, tuning, and export
python src/analysis_full.py

# Or run tuning & export using pre-selected pipeline
python src/model_tuning_and_export.py

# Generate temperature simulation dataset for Power BI
python src/generate_simulator.py
```

### 3. Open Power BI Dashboard
Open `dashboard/Power BI Dashboard.pbix` in Power BI Desktop to explore the interactive temperature simulator, actual lab data, model comparison, and feature importance pages.
