from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from feature_eng import copy_df
from feature_eng_old_df import old_df

# ---- HAM VERİ SETİ ----
print("####### HAM VERİ SETİ İÇİN CART + GRID SEARCH YAPILIYOR...")
X_old = old_df.drop(columns=['Sales'], axis=1)
y_old = old_df['Sales']

X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, y_old, test_size=0.2, random_state=42)

# Grid Search parametreleri
param_grid_cart = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_cart_old = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_cart, cv=4, scoring='r2', n_jobs=-1)
grid_cart_old.fit(X_train_old, y_train_old)

best_cart_old = grid_cart_old.best_estimator_
y_pred_old = best_cart_old.predict(X_test_old)

rmse_old = np.sqrt(mean_squared_error(y_test_old, y_pred_old))
mae_old = mean_absolute_error(y_test_old, y_pred_old)
r2_old = r2_score(y_test_old, y_pred_old)

print("####### HAM VERİ SETİ SONUÇLARI #########")
print("Best Params:", grid_cart_old.best_params_)
print("rmse:", rmse_old)
print("mae:", mae_old)
print("r2:", r2_old)


# ---- DEĞİŞTİRİLMİŞ VERİ SETİ ----
print("####### DEĞİŞTİRİLMİŞ VERİ SETİ İÇİN CART + GRID SEARCH YAPILIYOR...")
X_new = copy_df.drop(columns=['Sales'], axis=1)
y_new = copy_df['Sales']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

grid_cart_new = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_cart, cv=4, scoring='r2', n_jobs=-1)
grid_cart_new.fit(X_train_new, y_train_new)

best_cart_new = grid_cart_new.best_estimator_
y_pred_new = best_cart_new.predict(X_test_new)

rmse_new = np.sqrt(mean_squared_error(y_test_new, y_pred_new))
mae_new = mean_absolute_error(y_test_new, y_pred_new)
r2_new = r2_score(y_test_new, y_pred_new)

valid = cross_val_score(best_cart_new, X_new, y_new, cv=4, scoring='r2', n_jobs=1)

print("####### DEĞİŞEN VERİ SETİ SONUÇLARI #########")
print("Best Params:", grid_cart_new.best_params_)
print("rmse:", rmse_new)
print("mae:", mae_new)
print("r2:", r2_new)
print("cross_val r2:", valid)
