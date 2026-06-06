import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from feature_eng import copy_df
from feature_eng_old_df import old_df
from xgboost import XGBRegressor

# ---- HAM VERİ SETİ ----
print("####### HAM VERİ SETİ İÇİN XGBOOST + GRID SEARCH YAPILIYOR...")
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# --- HAM VERİ SETİ ---
X_old = old_df.drop(columns=['Sales'], axis=1)
y_old = old_df['Sales']

X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, y_old, test_size=0.2, random_state=42)

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

grid_xgb_old = GridSearchCV(
    XGBRegressor(random_state=42, objective='reg:squarederror'),
    param_grid_xgb, cv=4, scoring='r2', n_jobs=-1
)
grid_xgb_old.fit(X_train_old, y_train_old)

best_xgb_old = grid_xgb_old.best_estimator_
y_pred_old = best_xgb_old.predict(X_test_old)

rmse_old = np.sqrt(mean_squared_error(y_test_old, y_pred_old))
mae_old = mean_absolute_error(y_test_old, y_pred_old)
r2_old = r2_score(y_test_old, y_pred_old)

print("####### HAM VERİ SETİ SONUÇLARI #########")
print("Best Params:", grid_xgb_old.best_params_)
print("rmse:", rmse_old)
print("mae:", mae_old)
print("r2:", r2_old)

# --- SHAP ile HAM VERİ SETİ AÇIKLANABİLİRLİĞİ ---
X_sample_old = X_test_old.sample(200, random_state=42)
explainer_old = shap.TreeExplainer(best_xgb_old)
shap_values_old = explainer_old.shap_values(X_sample_old)

shap.summary_plot(shap_values_old, X_sample_old, plot_type="bar")
shap.summary_plot(shap_values_old, X_sample_old)

force_plot_old = shap.force_plot(
    explainer_old.expected_value,
    shap_values_old[0, :],
    X_sample_old.iloc[0, :]
)
shap.save_html("xgb_force_plot_old.html", force_plot_old)

# --- DEĞİŞTİRİLMİŞ VERİ SETİ ---
X_new = copy_df.drop(columns=['Sales'], axis=1)
y_new = copy_df['Sales']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

grid_xgb_new = GridSearchCV(
    XGBRegressor(random_state=42, objective='reg:squarederror'),
    param_grid_xgb, cv=4, scoring='r2', n_jobs=-1
)
grid_xgb_new.fit(X_train_new, y_train_new)

best_xgb_new = grid_xgb_new.best_estimator_
y_pred_new = best_xgb_new.predict(X_test_new)

rmse_new = np.sqrt(mean_squared_error(y_test_new, y_pred_new))
mae_new = mean_absolute_error(y_test_new, y_pred_new)
r2_new = r2_score(y_test_new, y_pred_new)

valid_new = cross_val_score(best_xgb_new, X_new, y_new, cv=4, scoring='r2', n_jobs=1)

print("####### DEĞİŞTİRİLMİŞ VERİ SETİ SONUÇLARI #########")
print("Best Params:", grid_xgb_new.best_params_)
print("rmse:", rmse_new)
print("mae:", mae_new)
print("r2:", r2_new)
print("cross_val r2:", valid_new)

# --- SHAP ile DEĞİŞTİRİLMİŞ VERİ SETİ AÇIKLANABİLİRLİĞİ ---
X_sample_new = X_test_new.sample(200, random_state=42)
explainer_new = shap.TreeExplainer(best_xgb_new)
shap_values_new = explainer_new.shap_values(X_sample_new)

shap.summary_plot(shap_values_new, X_sample_new, plot_type="bar")
shap.summary_plot(shap_values_new, X_sample_new)

force_plot_new = shap.force_plot(
    explainer_new.expected_value,
    shap_values_new[0, :],
    X_sample_new.iloc[0, :]
)
shap.save_html("xgb_force_plot_new.html", force_plot_new)

