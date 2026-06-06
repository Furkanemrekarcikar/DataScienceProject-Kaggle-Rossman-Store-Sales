import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.EDA import df
import pandas as pd

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

df = df.dropna(subset=['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'])
df.head(5)
print(df.shape)
df.nunique()

old_df = df.copy()
old_df['Promo2SinceWeek'] = old_df['Promo2SinceWeek'].fillna(1)
old_df['Promo2SinceYear'] = old_df['Promo2SinceYear'].fillna(0)

old_df.head(5)

assort_dummies = pd.get_dummies(old_df["Assortment"], prefix='assort')
old_df = pd.concat([old_df, assort_dummies], axis=1)
old_df.drop(columns=["Assortment"], inplace=True)

strtype_dummies = pd.get_dummies(old_df["StoreType"], prefix='store')
old_df = pd.concat([old_df, strtype_dummies], axis=1)
old_df.drop(columns=["StoreType"], inplace=True)

sthol_dummies = pd.get_dummies(old_df["StateHoliday"], prefix='state')
old_df = pd.concat([old_df, sthol_dummies], axis=1)
old_df.drop(columns=["StateHoliday"], inplace=True)

old_df.drop(columns=["Customers_log","Date"], inplace=True)
old_df.head(5)
old_df.shape
