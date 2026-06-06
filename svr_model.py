from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from feature_eng import copy_df
from feature_eng_old_df import old_df

# ---- HAM VERİ SETİ ----
print("####### HAM VERİ SETİ İÇİN SVR YAPILIYOR...")
X_old = old_df.drop(columns=['Sales'], axis=1)
y_old = old_df['Sales']

X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, y_old, test_size=0.2, random_state=42)

svr_model_old = SVR().fit(X_train_old, y_train_old)
y_pred_old = svr_model_old.predict(X_test_old)

rmse_old = np.sqrt(mean_squared_error(y_test_old, y_pred_old))
mae_old = mean_absolute_error(y_test_old, y_pred_old)
r2_old = r2_score(y_test_old, y_pred_old)

print("####### HAM VERİ SETİ SONUÇLARI #########")
print("rmse:", rmse_old)
print("mae:", mae_old)
print("r2:", r2_old)


# ---- DEĞİŞTİRİLMİŞ VERİ SETİ ----
print("####### DEĞİŞTİRİLMİŞ VERİ SETİ İÇİN SVR YAPILIYOR...")
X_new = copy_df.drop(columns=['Sales'], axis=1)
y_new = copy_df['Sales']

X_train_new, X_test_new, y_train_new, y_test_new = (
    train_test_split(X_new, y_new, test_size=0.2, random_state=42))
svr_model_new = SVR().fit(X_train_new, y_train_new)
y_pred_new = svr_model_new.predict(X_test_new)

rmse_new = np.sqrt(mean_squared_error(y_test_new, y_pred_new))
mae_new = mean_absolute_error(y_test_new, y_pred_new)
r2_new = r2_score(y_test_new, y_pred_new)

valid = cross_val_score(svr_model_new, X_new, y_new, cv=4, scoring='r2', n_jobs=1)

print("####### DEĞİŞEN VERİ SETİ SONUÇLARI #########")
print("rmse:", rmse_new)
print("mae:", mae_new)
print("r2:", r2_new)
print("cross_val r2:", valid)