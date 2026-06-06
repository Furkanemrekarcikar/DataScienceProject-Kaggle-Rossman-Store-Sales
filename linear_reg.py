from feature_eng import copy_df
from test_df import test_df_final
from feature_eng_old_df import old_df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

print("####### HAM VERİ SETİ İÇİN LİNEER REGRESYON YAPILIYOR...")
X_old = old_df.drop(columns=['Sales'], axis=1)
y_old = old_df['Sales']

X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, y_old, test_size=0.2, random_state=42)

linear_reg_model = LinearRegression().fit(X_train_old, y_train_old)
y_pred_old = linear_reg_model.predict(X_test_old)

rmse = np.sqrt(mean_squared_error(y_test_old,y_pred_old))
mae = mean_absolute_error(y_test_old,y_pred_old)
r2 = r2_score(y_test_old,y_pred_old)

print("####### HAM VERİ SETİ SONUÇLARI #########")
print("rmse:" , rmse)  # 1211
print("mae:", mae)     # 870
print("r2:", r2)       # 0.9009

print("######### DEĞİŞTİRİLMİŞ VERİ SETİ İÇİN LİNEER REGRESYON YAPILIYOR...")

X_new = copy_df.drop(columns=['Sales'], axis=1)
y_new = copy_df['Sales']
X_train_new, X_test_new, y_train_new, y_test_new = (
    train_test_split(X_new, y_new, test_size=0.2, random_state=42))
linear_reg_model_new = LinearRegression().fit(X_new, y_new)
y_pred_new = linear_reg_model_new.predict(X_test_new)

rmse_new = np.sqrt(mean_squared_error(y_test_new,y_pred_new))
mae_new = mean_absolute_error(y_test_new,y_pred_new)
r2_new = r2_score(y_test_new,y_pred_new)

print("####### DEĞİŞEN VERİ SETİ SONUÇLARI #########")
print("rmse:" , rmse_new)  # 1631
print("mae:", mae_new)     # 1154
print("r2:", r2_new)       # 0.82
