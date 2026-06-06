# Rossmann Store Sales Prediction

A machine learning project that predicts daily sales for Rossmann drug store branches using historical sales data and store metadata.

## Dataset

The project uses the [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) dataset:

| File | Description |
|------|-------------|
| `datasets/train.csv` | Historical sales data with features and target (`Sales`) |
| `datasets/test.csv` | Test set for prediction |
| `datasets/store.csv` | Supplemental store metadata (type, assortment, competition info) |

## Project Structure

```
internProject2/
├── datasets/
│   ├── EDA.py              # Exploratory data analysis and data merging
│   ├── train.csv
│   ├── test.csv
│   └── store.csv
├── store_analysis.py       # Store-level EDA (correlation, distributions)
├── feature_eng.py          # Feature engineering on merged dataset
├── feature_eng_old_df.py   # Feature engineering on raw dataset (baseline)
├── test_df.py              # Test set preparation
├── linear_reg.py           # Linear Regression model
├── knn_model.py            # K-Nearest Neighbors model
├── svr_model.py            # Support Vector Regression model
├── cart.py                 # Decision Tree (CART) model
├── random_forest.py        # Random Forest model with GridSearchCV
└── xgboost.py              # XGBoost model with GridSearchCV + SHAP
```

## Pipeline

1. **EDA** (`datasets/EDA.py`, `store_analysis.py`) — merges train and store data, explores distributions, detects outliers via IQR, visualizes correlations.

2. **Feature Engineering** (`feature_eng.py`) — constructs new features:
   - `Promo2DurationWeeks`: how long the store has been in the recurring promo
   - `CompetitionImpact`: competition proximity weighted by how long it has been open
   - `Store_Avg_Sales`, `Store_Std_Sales`, etc.: per-store aggregated statistics
   - One-hot encoding for `DayOfWeek`, `StoreType`, `Assortment`, `StateHoliday`

3. **Modeling** — each model script trains on both the raw (`old_df`) and engineered (`copy_df`) datasets and reports RMSE, MAE, and R² for comparison. XGBoost additionally generates SHAP explanations.

## Models & Results

Each model is evaluated with an 80/20 train-test split. Grid search with 4-fold cross-validation is applied where noted.

| Model | Dataset | Notes |
|-------|---------|-------|
| Linear Regression | raw + engineered | Baseline, R² ≈ 0.90 on raw |
| KNN | raw + engineered | — |
| SVR | raw + engineered | — |
| CART | raw + engineered | — |
| Random Forest | raw + engineered | GridSearchCV |
| XGBoost | raw + engineered | GridSearchCV + SHAP |

SHAP force plots are saved to `xgb_force_plot_old.html` and `xgb_force_plot_new.html`.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

## Usage

Run scripts from the project root in order:

```bash
python datasets/EDA.py        # loads and preprocesses data
python store_analysis.py      # store-level visualizations
python feature_eng.py         # builds engineered feature set
python xgboost.py             # train best model + SHAP analysis
```
