from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from feature_eng import copy_df
from feature_eng_old_df import old_df
import numpy as np

# ---- HAM VERİ SETİ ----
print("####### HAM VERİ SETİ İÇİN RANDOM FOREST + GRID SEARCH YAPILIYOR...")
X_old = old_df.drop(columns=['Sales'], axis=1)
y_old = old_df['Sales']

X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, y_old, test_size=0.2, random_state=42)

# Grid Search parametreleri
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_rf_old = GridSearchCV(RandomForestRegressor(random_state=42),
                           param_grid_rf, cv=4, scoring='r2', n_jobs=-1)
grid_rf_old.fit(X_train_old, y_train_old)

best_rf_old = grid_rf_old.best_estimator_
y_pred_old = best_rf_old.predict(X_test_old)

rmse_old = np.sqrt(mean_squared_error(y_test_old, y_pred_old))
mae_old = mean_absolute_error(y_test_old, y_pred_old)
r2_old = r2_score(y_test_old, y_pred_old)

print("####### HAM VERİ SETİ SONUÇLARI #########")
print("Best Params:", grid_rf_old.best_params_)
print("rmse:", rmse_old)
print("mae:", mae_old)
print("r2:", r2_old)


# ---- DEĞİŞTİRİLMİŞ VERİ SETİ ----
print("####### DEĞİŞTİRİLMİŞ VERİ SETİ İÇİN RANDOM FOREST + GRID SEARCH YAPILIYOR...")
X_new = copy_df.drop(columns=['Sales'], axis=1)
y_new = copy_df['Sales']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

grid_rf_new = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=4, scoring='r2', n_jobs=-1)
grid_rf_new.fit(X_train_new, y_train_new)

best_rf_new = grid_rf_new.best_estimator_
y_pred_new = best_rf_new.predict(X_test_new)

rmse_new = np.sqrt(mean_squared_error(y_test_new, y_pred_new))
mae_new = mean_absolute_error(y_test_new, y_pred_new)
r2_new = r2_score(y_test_new, y_pred_new)

valid = cross_val_score(best_rf_new, X_new, y_new, cv=4, scoring='r2', n_jobs=1)

print("####### DEĞİŞEN VERİ SETİ SONUÇLARI #########")
print("Best Params:", grid_rf_new.best_params_)
print("rmse:", rmse_new)
print("mae:", mae_new)
print("r2:", r2_new)
print("cross_val r2:", valid)
