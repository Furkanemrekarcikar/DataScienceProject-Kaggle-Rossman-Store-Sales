import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.EDA import store_df, test_df, train_df
import pandas as pd
from feature_eng import store_stats, copy_df

test_df.head(5)
store_df.head(5)
test_df_final = pd.merge(test_df, store_df, how='left', on='Store')
test_df_final.head(5)
test_df_final.isnull().sum()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


test_df_final['Date'] = pd.to_datetime(test_df_final['Date'])
test_df_final['StateHoliday'] = test_df_final['StateHoliday'].astype(str).astype('category')

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

for month in months:
    test_df_final[f'Promo_{month}'] = test_df_final['PromoInterval'].apply(
        lambda x: int(month in str(x)) if pd.notnull(x) else 0)

test_df_final.drop(columns=['PromoInterval'], inplace=True)


num_cols = test_df_final.select_dtypes(include=['int64', 'float64'])

def corr_matrix(num_cols):
    correlation_matrix = num_cols.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelasyon Matrisi")
    plt.show()

for col in test_df_final.columns:
    if test_df_final[col].dtype == 'object':
        test_df_final[col] = test_df_final[col].astype('category')


month_mode = test_df_final["CompetitionOpenSinceMonth"].mode()[0]
year_mode = test_df_final["CompetitionOpenSinceYear"].mode()[0]

test_df_final.loc[
    test_df_final["CompetitionDistance"].notna() & test_df_final["CompetitionOpenSinceMonth"].isna(),
    "CompetitionOpenSinceMonth"] = month_mode

test_df_final.loc[
    test_df_final["CompetitionDistance"].notna() & test_df_final["CompetitionOpenSinceYear"].isna(),
    "CompetitionOpenSinceYear"] = year_mode

print(test_df_final.head())
print(test_df_final.dtypes)
print(test_df_final.isnull().sum())


test_df_final["Year"] = test_df_final["Date"].dt.year
test_df_final["Month"] = test_df_final["Date"].dt.month
test_df_final["Day"] = test_df_final["Date"].dt.day

test_df_final = test_df_final.dropna(subset=['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'])


test_df_final['Promo2SinceWeek'] = test_df_final['Promo2SinceWeek'].fillna(1)
test_df_final['Promo2SinceYear'] = test_df_final['Promo2SinceYear'].fillna(0)

test_df_final['Promo2Since'] = pd.to_datetime(
    test_df_final['Promo2SinceYear'].astype(int).astype(str)
    + '-W' + test_df_final['Promo2SinceWeek'].astype(int).astype(str) + '-1',
    format='%G-W%V-%u',
    errors='coerce')

test_df_final['Promo2DurationWeeks'] = ((test_df_final['Date'] - test_df_final['Promo2Since']).dt.days // 7).clip(lower=0).fillna(0).astype(int)
test_df_final.drop(columns=['Promo2SinceWeek', 'Promo2SinceYear', 'Promo2Since'], inplace=True)


day_dummies = pd.get_dummies(test_df_final['DayOfWeek'], prefix='Day')
test_df_final = pd.concat([test_df_final, day_dummies], axis=1)
test_df_final.drop(columns=["DayOfWeek"], inplace=True)


test_df_final["isWeekend"] = test_df_final["Date"].dt.dayofweek >= 5
test_df_final.head(5)


test_df_final['CompetitionSince'] = pd.to_datetime(
    test_df_final['CompetitionOpenSinceYear'].astype(int).astype(str) + '-' +
    test_df_final['CompetitionOpenSinceMonth'].astype(int).astype(str) + '-01',
    format='%Y-%m-%d',
    errors='coerce')

test_df_final['CompetitionDurationWeeks'] = ((test_df_final['Date'] - test_df_final['CompetitionSince']).dt.days // 7).clip(lower=0).fillna(0).astype(int)
test_df_final['CompetitionImpact'] = (test_df_final['CompetitionDurationWeeks'] / (test_df_final['CompetitionDistance'] + 1)) * test_df_final['Open']
test_df_final.drop(columns=["Date", "CompetitionSince", "CompetitionDurationWeeks",
                            "CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"], inplace=True)


test_df_final.drop(columns=["Year", "Month", "Day"], inplace=True)
test_df_final.head(5)

assort_dummies = pd.get_dummies(test_df_final["Assortment"], prefix='assort')
test_df_final = pd.concat([test_df_final, assort_dummies], axis=1)
test_df_final.drop(columns=["Assortment"], inplace=True)

strtype_dummies = pd.get_dummies(test_df_final["StoreType"], prefix='store')
test_df_final = pd.concat([test_df_final, strtype_dummies], axis=1)
test_df_final.drop(columns=["StoreType"], inplace=True)

sthol_dummies = pd.get_dummies(test_df_final["StateHoliday"], prefix='state')
test_df_final = pd.concat([test_df_final, sthol_dummies], axis=1)
test_df_final.drop(columns=["StateHoliday"], inplace=True)


temp = test_df_final["CompetitionDistance"]
test_df_final.drop(columns=["CompetitionDistance"], inplace=True)
test_df_final["CompetitionDistance"] = temp


print(test_df_final.head())
print(test_df_final.dtypes)
print(test_df_final.shape)

for col in store_stats.columns:
    if col != "Store":
        test_df_final[col] = test_df_final["Store"].map(store_stats.set_index("Store")[col])


for col in store_stats.columns:
    if col != "Store":
        median_value = copy_df[col].median()
        test_df_final[col] = test_df_final[col].fillna(median_value)

test_df_final["state_b"] = False
test_df_final["state_c"] = False


train_cols_except_sales = [col for col in copy_df.columns if col != 'Sales']
test_df_final = test_df_final[train_cols_except_sales]
if 'Id' in test_df.columns:
    test_df_final.insert(0, 'Id', test_df['Id'])

test_df_final.head(5)
test_df_final.shape
test_df_final.columns


