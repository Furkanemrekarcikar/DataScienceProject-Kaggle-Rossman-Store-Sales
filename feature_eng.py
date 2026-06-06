import numpy as np
from datasets.EDA import df, corr_matrix
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df.head(5)
df.dtypes

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

df = df.dropna(subset=['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'])
df.head(5)
print(df.shape)
df.nunique()

copy_df = df.copy()
copy_df['Promo2SinceWeek'] = copy_df['Promo2SinceWeek'].fillna(1)
copy_df['Promo2SinceYear'] = copy_df['Promo2SinceYear'].fillna(0)


copy_df['Promo2Since'] = pd.to_datetime(
    copy_df['Promo2SinceYear'].astype(int).astype(str)
    + '-W' + copy_df['Promo2SinceWeek'].astype(int).astype(str) + '-1',
    format='%G-W%V-%u',
    errors='coerce')
copy_df['Promo2DurationWeeks'] = ((copy_df['Date'] -
        copy_df['Promo2Since']).dt.days // 7).clip(lower=0).fillna(0).astype(int)

copy_df.drop(columns=['Promo2SinceWeek', 'Promo2SinceYear', 'Promo2Since'], inplace=True)
copy_df.head(5)

day_dummies = pd.get_dummies(copy_df['DayOfWeek'], prefix='Day')
copy_df = pd.concat([copy_df, day_dummies], axis=1)
copy_df.head(5)

copy_df.drop(columns=["DayOfWeek"], inplace=True)

copy_df["isWeekend"] = copy_df["Date"].dt.dayofweek >= 5
copy_df.head(5)

copy_df.drop(columns=["Date"], inplace=True)
copy_df.head(5)
copy_df.dtypes
copy_df.shape

copy_df["Promo2_Total_Impact"] = (copy_df["Customers"]
                                  * copy_df["Promo2DurationWeeks"])
scaler = MinMaxScaler()
copy_df["P2_TotalImp_Scaled"] = scaler.fit_transform(copy_df[["Promo2_Total_Impact"]])



copy_df.head(20)
copy_df.dtypes
copy_num_cols = copy_df.select_dtypes(include=['int64', 'float64','int32'])
corr_matrix(copy_num_cols)
copy_df.drop(columns=["Promo2_Total_Impact"], inplace=True)
copy_df[copy_df["Open"] == 0]

copy_df["PromoTotal"] = copy_df["Open"] * copy_df["Promo"] * copy_df["Customers"]
copy_df.head(100)
copy_num_cols = copy_df.select_dtypes(include=['int64', 'float64','int32'])


copy_df["Date"] = df["Date"]

copy_df.head(5)

copy_df['CompetitionSince'] = pd.to_datetime(
    copy_df['CompetitionOpenSinceYear'].astype(int).astype(str) + '-' +
    copy_df['CompetitionOpenSinceMonth'].astype(int).astype(str) + '-01',
    format='%Y-%m-%d',
    errors='coerce')


copy_df['CompetitionDurationWeeks'] = ((copy_df['Date'] - copy_df['CompetitionSince']).dt.days // 7).clip(lower=0).fillna(0).astype(int)

copy_df.head(3)
copy_df.drop(columns=["Date", "CompetitionSince"], inplace=True)
copy_df.head(5)

copy_df['CompetitionImpact'] = (copy_df['CompetitionDurationWeeks'] / (copy_df['CompetitionDistance'] + 1)) * copy_df['Open']
copy_df.head(5)
copy_num_cols = copy_df.select_dtypes(include=['int64', 'float64','int32',"float32"])
corr_matrix(copy_num_cols)
copy_df.drop(columns=["CompetitionDurationWeeks"], inplace=True)
copy_df.head(5)
copy_df.drop(columns=["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"], inplace=True)
copy_df.head(5)

copy_df.drop(columns=["Year", "Month", "Day"], inplace=True)

assort_dummies = pd.get_dummies(copy_df["Assortment"],prefix='assort')
copy_df = pd.concat([copy_df, assort_dummies], axis=1)
copy_df.head(5)
copy_df.drop(columns=["Assortment"], inplace=True)

strtype_dummies = pd.get_dummies(copy_df["StoreType"], prefix='store')
copy_df = pd.concat([copy_df, strtype_dummies], axis=1)
copy_df.drop(columns=["StoreType"], inplace=True)

sthol_dummies = pd.get_dummies(copy_df["StateHoliday"], prefix='state')
copy_df = pd.concat([copy_df, sthol_dummies], axis=1)
copy_df.head(5)
copy_df.drop(columns=["StateHoliday"], inplace=True)
copy_df.head(5)

temp = copy_df["CompetitionDistance"]
copy_df.drop(columns=["CompetitionDistance"], inplace=True)
copy_df.head(5)
copy_df["CompetitionDistance"] = temp
copy_df.head(5)

store_stats = copy_df.groupby("Store").agg(
    Store_Avg_Sales=("Sales", "mean"),
    Store_Median_Sales=("Sales", "median"),
    Store_Std_Sales=("Sales", "std"),
    Store_Promo_Rate=("Promo", "mean"),
    Store_Total_Sales=("Sales", "sum"),
    Store_Open_Days=("Open", "sum")
).reset_index()

copy_df.isnull().sum()
store_stats["Store_Std_Sales"].fillna(0, inplace=True)
copy_df = copy_df.merge(store_stats, on="Store", how="left")

print(copy_df[[ "Store", "Store_Avg_Sales", "Store_Promo_Rate"]].head())

copy_df.drop(columns=["Customers","Customers_log","PromoTotal","P2_TotalImp_Scaled"], inplace=True)

copy_df.head(5)
copy_df.shape
copy_df.columns