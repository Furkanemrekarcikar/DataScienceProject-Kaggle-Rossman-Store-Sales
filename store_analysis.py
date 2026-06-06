import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
store_df = pd.read_csv('datasets/store.csv', low_memory=False)
store_df.head(5)

print("----- Genel Bilgi -----")
print(store_df.info())
print("\n----- İlk 5 Satır -----")
print(store_df.head())
print("\n----- Eksik Veri Sayısı -----")
print(store_df.isnull().sum())

store_df.shape

month_mode_store = store_df["CompetitionOpenSinceMonth"].mode()[0]
year_mode_store = store_df["CompetitionOpenSinceYear"].mode()[0]

store_df.loc[store_df["CompetitionDistance"].notna() & store_df["CompetitionOpenSinceMonth"].isna(), "CompetitionOpenSinceMonth"] = month_mode_store
store_df.loc[store_df["CompetitionDistance"].notna() & store_df["CompetitionOpenSinceYear"].isna(), "CompetitionOpenSinceYear"] = year_mode_store
store_df = store_df.dropna(subset=['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'])

store_num_cols = store_df.select_dtypes(include=['int64', 'float64','int32'])

def corr_matrix(num_cols):
    correlation_matrix = num_cols.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelasyon Matrisi")
    plt.show()
corr_matrix(store_num_cols)


print(store_df["StoreType"].value_counts())
sns.countplot(x="StoreType", data=store_df)
plt.title("StoreType Dağılımı")
plt.show()

# Assortment
print(store_df["Assortment"].value_counts())
sns.countplot(x="Assortment", data=store_df)
plt.title("Assortment Dağılımı")
plt.show()

# Promo2
print(store_df["Promo2"].value_counts())
sns.countplot(x="Promo2", data=store_df)
plt.title("Promo2 Dağılımı")
plt.show()

print(store_df["CompetitionDistance"].describe())

# Histogram
plt.figure(figsize=(8,4))
sns.histplot(store_df["CompetitionDistance"], bins=50, kde=True)
plt.title("CompetitionDistance Dağılımı")
plt.show()

# Boxplot
plt.figure(figsize=(8,4))
sns.boxplot(x=store_df["CompetitionDistance"])
plt.title("CompetitionDistance Boxplot")
plt.show()

print(store_df["CompetitionOpenSinceYear"].value_counts().sort_index())
store_df["CompetitionOpenSinceYear"].dropna().astype(int).hist(bins=20)
plt.title("CompetitionOpenSinceYear Dağılımı")
plt.show()

# Promo2SinceYear
print(store_df["Promo2SinceYear"].value_counts().sort_index())
store_df["Promo2SinceYear"].dropna().astype(int).hist(bins=20)
plt.title("Promo2SinceYear Dağılımı")
plt.show()

# StoreType'a göre CompetitionDistance
print(store_df.groupby("StoreType")["CompetitionDistance"].mean())

sns.boxplot(x="StoreType", y="CompetitionDistance", data=store_df)
plt.title("CompetitionDistance vs StoreType")
plt.show()

# Promo2'ye göre CompetitionDistance
print(store_df.groupby("Promo2")["CompetitionDistance"].mean())