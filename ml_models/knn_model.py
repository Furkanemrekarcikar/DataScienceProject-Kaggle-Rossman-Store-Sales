from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from feature_eng import copy_df
from feature_eng_old_df import old_df
import numpy as np

print("####### HAM VERİ SETİ İÇİN KNN YAPILIYOR...")
X_old = old_df.drop(columns=['Sales'], axis=1)
y_old = old_df['Sales']

X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, y_old, test_size=0.2, random_state=42)

knn_model_old = KNeighborsRegressor().fit(X_train_old, y_train_old)
y_pred_old = knn_model_old.predict(X_test_old)

rmse_old = np.sqrt(root_mean_squared_error(y_test_old, y_pred_old))
mae_old = mean_absolute_error(y_test_old, y_pred_old)
r2_old = r2_score(y_test_old, y_pred_old)

print("####### HAM VERİ SETİ SONUÇLARI #########")
print("rmse:" , rmse_old)  # 24.64
print("mae:", mae_old)     # 377
print("r2:", r2_old)       # 0.97

print("######### DEĞİŞTİRİLMİŞ VERİ SETİ İÇİN KNN YAPILIYOR...")

X_new = copy_df.drop(columns=['Sales'], axis=1)
y_new = copy_df['Sales']
X_train_new, X_test_new, y_train_new, y_test_new =\
    train_test_split(X_new, y_new, test_size=0.2, random_state=42)
knn_model_new = KNeighborsRegressor().fit(X_train_new, y_train_new)
y_pred_new = knn_model_new.predict(X_test_new)

rmse_new = np.sqrt(mean_squared_error(y_test_new,y_pred_new))
mae_new = mean_absolute_error(y_test_new,y_pred_new)
r2_new = r2_score(y_test_new,y_pred_new)

valid = cross_val_score(knn_model_new, X_new, y_new, cv=4, scoring='r2',n_jobs=1)
valid    # array([0.63638691, 0.68241838, 0.59805443, 0.63759227])
print("####### DEĞİŞEN VERİ SETİ SONUÇLARI #########")
print("rmse:" , rmse_new)  # 1648
print("mae:", mae_new)     # 1054
print("r2:", r2_new)       # 0.81
