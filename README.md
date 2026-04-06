# Predicting Lattice Thermal Conductivity with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Predicting lattice thermal conductivity (κ_L) of crystalline materials from **crystal structure, composition, and electronic properties** using a progressive ML pipeline - no elastic moduli, no Debye temperatures, no derived thermal quantities.

---

## Overview

Lattice thermal conductivity governs heat transport in solids and is critical for:

- **Thermoelectrics** : low κ_L maximises ZT = σS²T / κ
- **Thermal management** : heat sinks, substrates, packaging
- **Thermal barrier coatings** : aerospace and energy applications

First-principles calculations (solving the Boltzmann Transport Equation) are computationally expensive. This project trains ML models on **5,530 DFT-computed compounds** from the [AFLOW database](http://www.aflowlib.org/), enabling rapid κ_L screening without phonon calculations.

---

## The Leakage Problem (and How We Fixed It)

The AFLOW database computes κ_L via the **Slack model** inside its AGL pipeline. Several commonly-used features are co-outputs of the same calculation including them leaks the answer into the features:

| Feature group | Why it's leaky |
|---|---|
| `agl_acoustic_debye`, `agl_gruneisen`, `agl_heat_capacity_*` | Same Debye–Grüneisen model that computes κ_L |
| `ael_speed_sound_*`, `ael_debye_temperature` | Appear directly in Slack formula: κ_L ∝ v³ |
| `ael_bulk/shear/youngs_modulus_*`, `ael_poisson_ratio` | Upstream Slack inputs: κ_L ∝ E^(3/2) — linear regression alone hits R²≈0.92 |
| `scintillation_attenuation_length` | Despite the name, in AFLOW this is the **phonon mean free path** λ; κ_L = ⅓·C·v·λ |

After removing all leaky features, we train on genuinely independent structural and electronic descriptors.

---

## Results

| Model | R² (test) | MAE (W/mK) | RMSE (W/mK) |
|---|---|---|---|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| **LightGBM** | **—** | **—** | **—** |
| **LightGBM Top-10** | **—** | **—** | **—** |

> Run the notebook to populate exact numbers.

---

## Project Structure

```
├── thermal_conductivity_prediction.ipynb   ← main notebook
├── thermal_conductivity_dataset.csv        ← AFLOW dataset (add manually)
├── kl_predictor_lgbm.pkl                   ← saved LightGBM model
├── kl_top10_features.pkl                   ← feature list for inference
├── target_distribution.png                 ← raw vs log-transformed κ_L
├── shap_summary.png                        ← SHAP feature importance plot
└── README.md
```

---

## Notebook Walkthrough

| Section | Description |
|---|---|
| **1. Setup & Data Loading** | Load 5,530-compound AFLOW dataset (173 features) |
| **2. Feature Engineering** | Two-pass reduction: 173 → 88 → 52 columns |
| **3. Data Quality Check** | Missing values, dtype breakdown, cardinality audit |
| **4. Removing Leaky Features** | Drop AGL/AEL/phonon-MFP features; 52 → ~22 clean features |
| **5. Encoding & Split** | One-hot encoding, 80/10/10 train/val/test split |
| **6. Target Distribution** | Visualise raw vs log-transformed κ_L |
| **7. Model Training** | Linear Regression → Random Forest → XGBoost → LightGBM |
| **8. Model Comparison** | Side-by-side R², MAE, RMSE on test set |
| **9. SHAP Feature Importance** | Which structural features drive predictions |
| **10. Lean Model** | Retrain LightGBM on top-10 SHAP features |
| **11. Save Model** | Serialise model + feature list with joblib |

---

## Installation

```bash
pip install numpy pandas scikit-learn lightgbm xgboost shap matplotlib joblib
```

---

## Usage

### Run the notebook

```bash
jupyter notebook thermal_conductivity_prediction.ipynb
```

### Load the saved model

```python
import joblib, numpy as np

model    = joblib.load('kl_predictor_lgbm.pkl')
features = joblib.load('kl_top10_features.pkl')

# your_df must contain the 10 required feature columns
kl_pred = np.expm1(model.predict(your_df[features]))
print(f'Predicted κ_L: {kl_pred[0]:.3f} W/mK')
```

---

## Dataset

The dataset is sourced from the AFLOW Materials Database. Download and place `thermal_conductivity_dataset.csv` in the project root before running the notebook.

- **Source:** [AFLOW](http://www.aflowlib.org/)
- **Samples:** 5,530 crystalline compounds
- **Target:** `thermal_conductivity_target` (W/mK)

---

## References

- Curtarolo et al., *AFLOW: An automatic framework for high-throughput materials discovery*, Comput. Mater. Sci. (2012)
- Slack, G.A., *Solid State Physics*, Vol. 34 (1979) — Slack model for κ_L
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS (2017) — SHAP

---

## License

MIT License
