import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 3000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.2f}'.format)        # Satır genişliğini artır
pd.set_option('display.max_colwidth', None) # Hücre içeriği sınırı kaldır


train_df = pd.read_csv('datasets/train.csv', low_memory=False)
test_df = pd.read_csv('datasets/test.csv',low_memory=False)
store_df = pd.read_csv('datasets/store.csv', low_memory=False)

train_df.shape
df = pd.merge(train_df, store_df, how='left', on='Store')
df.set_index('Store',inplace=True)
df.head(4)

for col in df.columns:
    print(f"{col}: {df[col].value_counts()}")

df.describe()

df['Date'] = pd.to_datetime(df['Date'])
df['StateHoliday'] = df['StateHoliday'].astype(str)
df['StateHoliday'] = df['StateHoliday'].astype('category')

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

for month in months:
    df[f'Promo_{month}'] = df['PromoInterval'].apply(
        lambda x: int(month in str(x)) if pd.notnull(x) else 0)
df.columns
df.drop(columns=['PromoInterval'], inplace=True)
df.head(5)
df.dtypes
num_cols = df.select_dtypes(include=['int64', 'float64'])
def corr_matrix(num_cols):
    correlation_matrix = num_cols.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelasyon Matrisi")
    plt.show()
corr_matrix(num_cols)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')

df.dtypes
df.isnull().sum()


month_mode = df["CompetitionOpenSinceMonth"].mode()[0]
year_mode = df["CompetitionOpenSinceYear"].mode()[0]

df.loc[df["CompetitionDistance"].notna() & df["CompetitionOpenSinceMonth"].isna(), "CompetitionOpenSinceMonth"] = month_mode
df.loc[df["CompetitionDistance"].notna() & df["CompetitionOpenSinceYear"].isna(), "CompetitionOpenSinceYear"] = year_mode

df.isnull().sum()

def detect_outliers_iqr(df, columns):
    outlier_info = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            'outlier_count': outliers.shape[0],
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    return outlier_info

out_table = detect_outliers_iqr(df,num_cols)
pd.DataFrame(out_table).T

df['Customers_log'] = np.log1p(df['Customers'])

num_cols = df.select_dtypes(include=['int64', 'float64'])
num_cols.columns
table = detect_outliers_iqr(df,num_cols)
pd.DataFrame(table).T

for col in num_cols.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), bins=30, kde=True)
    plt.title(f"Histogram - {col}")
    plt.xlabel(col)
    plt.ylabel("Frekans")
    plt.show()

df[df["Store"] == 45]
df.head(5)

