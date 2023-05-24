import gc
import pickle

import numpy as np
import pandas as pd
import sklearn
import torch
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer, MissingValuesFiller, InvertibleMapper
from darts.models import MovingAverage
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn import preprocessing
from tqdm import tqdm

# %matplotlib inline
torch.manual_seed(1)
np.random.seed(1)  # for reproducibility

# Load all Datasets
df_train = pd.read_csv('data/train.csv')
df_holidays_events = pd.read_csv('data/holidays_events.csv')
df_oil = pd.read_csv('data/oil.csv')
df_stores = pd.read_csv('data/stores.csv')
df_transactions = pd.read_csv('data/transactions.csv')
df_sample_submission = pd.read_csv('data/sample_submission.csv')

# Sales Data (Target)
family_list = df_train['family'].unique()

store_list = df_stores['store_nbr'].unique()

train_merged = pd.merge(df_train, df_stores, on='store_nbr')
train_merged = train_merged.sort_values(["store_nbr", "family", "date"])
train_merged = train_merged.astype({"store_nbr": 'str', "family": 'str', "city": 'str',
                                    "state": 'str', "type": 'str', "cluster": 'str'})

# Create TimeSeries objects (Darts) and arrange in a Dictionary clustered by Product Family

family_TS_dict = {}

for family in family_list:
    df_family = train_merged.loc[train_merged['family'] == family]

    list_of_TS_family = TimeSeries.from_group_dataframe(
        df_family,
        time_col="date",
        group_cols=["store_nbr", "family"],  # individual time series are extracted by grouping `df` by `group_cols`
        static_cols=["city", "state", "type", "cluster"],  # also extract these additional columns as static covariates
        value_cols="sales",  # target variable
        fill_missing_dates=True,
        freq='D')
    for ts in list_of_TS_family:
        ts = ts.astype(np.float32)

    list_of_TS_family = sorted(list_of_TS_family, key=lambda ts: int(ts.static_covariates_values()[0, 0]))
    family_TS_dict[family] = list_of_TS_family

# Transform the Sales Data

family_pipeline_dict = {}
family_TS_transformed_dict = {}

for key in family_TS_dict:
    train_filler = MissingValuesFiller(verbose=False, n_jobs=-1, name="Fill NAs")
    static_cov_transformer = StaticCovariatesTransformer(verbose=False,
                                                         transformer_cat=sklearn.preprocessing.OneHotEncoder(),
                                                         name="Encoder")  # OneHotEncoder would be better but takes longer
    log_transformer = InvertibleMapper(np.log1p, np.expm1, verbose=False, n_jobs=-1, name="Log-Transform")
    train_scaler = Scaler(verbose=False, n_jobs=-1, name="Scaling")

    train_pipeline = Pipeline([train_filler,
                               static_cov_transformer,
                               log_transformer,
                               train_scaler])

    training_transformed = train_pipeline.fit_transform(family_TS_dict[key])
    family_pipeline_dict[key] = train_pipeline
    family_TS_transformed_dict[key] = training_transformed

# Create TimeSeries objects (Darts) 1782

list_of_TS = TimeSeries.from_group_dataframe(
    train_merged,
    time_col="date",
    group_cols=["store_nbr", "family"],  # individual time series are extracted by grouping `df` by `group_cols`
    static_cols=["city", "state", "type", "cluster"],  # also extract these additional columns as static covariates
    value_cols="sales",  # target variable
    fill_missing_dates=True,
    freq='D')
for ts in list_of_TS:
    ts = ts.astype(np.float32)

list_of_TS = sorted(list_of_TS, key=lambda ts: int(ts.static_covariates_values()[0, 0]))

# Transform the Sales Data

train_filler = MissingValuesFiller(verbose=False, n_jobs=-1, name="Fill NAs")
static_cov_transformer = StaticCovariatesTransformer(verbose=False,
                                                     transformer_cat=sklearn.preprocessing.OneHotEncoder(),
                                                     name="Encoder")  # OneHotEncoder would be better but takes longer
log_transformer = InvertibleMapper(np.log1p, np.expm1, verbose=False, n_jobs=-1, name="Log-Transform")
train_scaler = Scaler(verbose=False, n_jobs=-1, name="Scaling")

train_pipeline = Pipeline([train_filler,
                           static_cov_transformer,
                           log_transformer,
                           train_scaler])

training_transformed = train_pipeline.fit_transform(list_of_TS)

# Create 7-day and 28-day moving average of sales

sales_moving_average_7 = MovingAverage(window=7)
sales_moving_average_28 = MovingAverage(window=28)

sales_moving_averages_dict = {}

for key in family_TS_transformed_dict:
    sales_mas_family = []

    for ts in family_TS_transformed_dict[key]:
        ma_7 = sales_moving_average_7.filter(ts)
        ma_7 = TimeSeries.from_series(ma_7.pd_series())
        ma_7 = ma_7.astype(np.float32)
        ma_7 = ma_7.with_columns_renamed(col_names=ma_7.components, col_names_new="sales_ma_7")
        ma_28 = sales_moving_average_28.filter(ts)
        ma_28 = TimeSeries.from_series(ma_28.pd_series())
        ma_28 = ma_28.astype(np.float32)
        ma_28 = ma_28.with_columns_renamed(col_names=ma_28.components, col_names_new="sales_ma_28")
        mas = ma_7.stack(ma_28)
        sales_mas_family.append(mas)

    sales_moving_averages_dict[key] = sales_mas_family

# General Covariates (Time-Based and Oil)

full_time_period = pd.date_range(start='2013-01-01', end='2017-08-31', freq='D')

# Time-Based Covariates

year = datetime_attribute_timeseries(time_index=full_time_period, attribute="year")
month = datetime_attribute_timeseries(time_index=full_time_period, attribute="month")
day = datetime_attribute_timeseries(time_index=full_time_period, attribute="day")
dayofyear = datetime_attribute_timeseries(time_index=full_time_period, attribute="dayofyear")
weekday = datetime_attribute_timeseries(time_index=full_time_period, attribute="dayofweek")
weekofyear = datetime_attribute_timeseries(time_index=full_time_period, attribute="weekofyear")
timesteps = TimeSeries.from_times_and_values(times=full_time_period,
                                             values=np.arange(len(full_time_period)),
                                             columns=["linear_increase"])

time_cov = year.stack(month).stack(day).stack(dayofyear).stack(weekday).stack(weekofyear).stack(timesteps)
time_cov = time_cov.astype(np.float32)

# Transform
time_cov_scaler = Scaler(verbose=False, n_jobs=-1, name="Scaler")
time_cov_train, time_cov_val = time_cov.split_before(pd.Timestamp('20170816'))
time_cov_scaler.fit(time_cov_train)
time_cov_transformed = time_cov_scaler.transform(time_cov)

# Oil Price

oil = TimeSeries.from_dataframe(df_oil,
                                time_col='date',
                                value_cols=['dcoilwtico'],
                                freq='D')

oil = oil.astype(np.float32)

# Transform
oil_filler = MissingValuesFiller(verbose=False, n_jobs=-1, name="Filler")
oil_scaler = Scaler(verbose=False, n_jobs=-1, name="Scaler")
oil_pipeline = Pipeline([oil_filler, oil_scaler])
oil_transformed = oil_pipeline.fit_transform(oil)

# Moving Averages for Oil Price
oil_moving_average_7 = MovingAverage(window=7)
oil_moving_average_28 = MovingAverage(window=28)

oil_moving_averages = []

ma_7 = oil_moving_average_7.filter(oil_transformed).astype(np.float32)
ma_7 = ma_7.with_columns_renamed(col_names=ma_7.components, col_names_new="oil_ma_7")
ma_28 = oil_moving_average_28.filter(oil_transformed).astype(np.float32)
ma_28 = ma_28.with_columns_renamed(col_names=ma_28.components, col_names_new="oil_ma_28")
oil_moving_averages = ma_7.stack(ma_28)

# Stack General Covariates Together

general_covariates = time_cov_transformed.stack(oil_transformed).stack(oil_moving_averages)

# Store-Specific Covariates (Transactions and Holidays)

# Transactions
df_transactions.sort_values(["store_nbr", "date"], inplace=True)

TS_transactions_list = TimeSeries.from_group_dataframe(
    df_transactions,
    time_col="date",
    group_cols=["store_nbr"],  # individual time series are extracted by grouping `df` by `group_cols`
    value_cols="transactions",
    fill_missing_dates=True,
    freq='D')

transactions_list = []

for ts in TS_transactions_list:
    series = TimeSeries.from_series(
        ts.pd_series())  # necessary workaround to remove static covariates (so I can stack covariates later on)
    series = series.astype(np.float32)
    transactions_list.append(series)

transactions_list[24] = transactions_list[24].slice(start_ts=pd.Timestamp('20130102'), end_ts=pd.Timestamp('20170815'))

from datetime import timedelta

transactions_list_full = []

for ts in transactions_list:
    if ts.start_time() > pd.Timestamp('20130101'):
        end_time = (ts.start_time() - timedelta(days=1))
        delta = end_time - pd.Timestamp('20130101')
        zero_series = TimeSeries.from_times_and_values(
            times=pd.date_range(start=pd.Timestamp('20130101'),
                                end=end_time, freq="D"),
            values=np.zeros(delta.days + 1))
        ts = zero_series.append(ts)
        transactions_list_full.append(ts)

transactions_filler = MissingValuesFiller(verbose=False, n_jobs=-1, name="Filler")
transactions_scaler = Scaler(verbose=False, n_jobs=-1, name="Scaler")

transactions_pipeline = Pipeline([transactions_filler, transactions_scaler])
transactions_transformed = transactions_pipeline.fit_transform(transactions_list_full)

# Moving Averages for Transactions
trans_moving_average_7 = MovingAverage(window=7)
trans_moving_average_28 = MovingAverage(window=28)

transactions_covs = []

for ts in transactions_transformed:
    ma_7 = trans_moving_average_7.filter(ts).astype(np.float32)
    ma_7 = ma_7.with_columns_renamed(col_names=ma_7.components, col_names_new="transactions_ma_7")
    ma_28 = trans_moving_average_28.filter(ts).astype(np.float32)
    ma_28 = ma_28.with_columns_renamed(col_names=ma_28.components, col_names_new="transactions_ma_28")
    trans_and_mas = ts.with_columns_renamed(col_names=ts.components, col_names_new="transactions").stack(ma_7).stack(
        ma_28)
    transactions_covs.append(trans_and_mas)

# Re-Defining Categories of Holidays in a Meaningful Way

df_holidays_events['type'] = np.where(df_holidays_events['transferred'] == True, 'Transferred',
                                      df_holidays_events['type'])

df_holidays_events['type'] = np.where(df_holidays_events['type'] == 'Transfer', 'Holiday',
                                      df_holidays_events['type'])

df_holidays_events['type'] = np.where(df_holidays_events['type'] == 'Additional', 'Holiday',
                                      df_holidays_events['type'])

df_holidays_events['type'] = np.where(df_holidays_events['type'] == 'Bridge', 'Holiday',
                                      df_holidays_events['type'])


# Assign Holidays to all TimeSeries and Save in Dictionary

def holiday_list(df_stores):
    listofseries = []

    for i in range(0, len(df_stores)):
        df_holiday_dummies = pd.DataFrame(columns=['date'])
        df_holiday_dummies["date"] = df_holidays_events["date"]

        df_holiday_dummies["national_holiday"] = np.where(
            ((df_holidays_events["type"] == "Holiday") & (df_holidays_events["locale"] == "National")), 1, 0)

        df_holiday_dummies["earthquake_relief"] = np.where(
            df_holidays_events['description'].str.contains('Terremoto Manabi'), 1, 0)

        df_holiday_dummies["christmas"] = np.where(df_holidays_events['description'].str.contains('Navidad'), 1, 0)

        df_holiday_dummies["football_event"] = np.where(df_holidays_events['description'].str.contains('futbol'), 1, 0)

        df_holiday_dummies["national_event"] = np.where(((df_holidays_events["type"] == "Event") & (
                df_holidays_events["locale"] == "National") & (~df_holidays_events['description'].str.contains(
            'Terremoto Manabi')) & (~df_holidays_events['description'].str.contains('futbol'))), 1, 0)

        df_holiday_dummies["work_day"] = np.where((df_holidays_events["type"] == "Work Day"), 1, 0)

        df_holiday_dummies["local_holiday"] = np.where(((df_holidays_events["type"] == "Holiday") & (
                (df_holidays_events["locale_name"] == df_stores['state'][i]) | (
                df_holidays_events["locale_name"] == df_stores['city'][i]))), 1, 0)

        listofseries.append(df_holiday_dummies)

    return listofseries


def remove_0_and_duplicates(holiday_list):
    listofseries = []

    for i in range(0, len(holiday_list)):
        df_holiday_per_store = list_of_holidays_per_store[i].set_index('date')

        df_holiday_per_store = df_holiday_per_store.loc[~(df_holiday_per_store == 0).all(axis=1)]

        df_holiday_per_store = df_holiday_per_store.groupby('date').agg(
            {'national_holiday': 'max', 'earthquake_relief': 'max',
             'christmas': 'max', 'football_event': 'max',
             'national_event': 'max', 'work_day': 'max',
             'local_holiday': 'max'}).reset_index()

        listofseries.append(df_holiday_per_store)

    return listofseries


def holiday_TS_list_54(holiday_list):
    listofseries = []

    for i in range(0, 54):
        holidays_TS = TimeSeries.from_dataframe(list_of_holidays_per_store[i],
                                                time_col='date',
                                                fill_missing_dates=True,
                                                fillna_value=0,
                                                freq='D')

        holidays_TS = holidays_TS.slice(pd.Timestamp('20130101'), pd.Timestamp('20170831'))
        holidays_TS = holidays_TS.astype(np.float32)
        listofseries.append(holidays_TS)

    return listofseries


list_of_holidays_per_store = holiday_list(df_stores)
list_of_holidays_per_store = remove_0_and_duplicates(list_of_holidays_per_store)
list_of_holidays_store = holiday_TS_list_54(list_of_holidays_per_store)

holidays_filler = MissingValuesFiller(verbose=False, n_jobs=-1, name="Filler")
holidays_scaler = Scaler(verbose=False, n_jobs=-1, name="Scaler")

holidays_pipeline = Pipeline([holidays_filler, holidays_scaler])
holidays_transformed = holidays_pipeline.fit_transform(list_of_holidays_store)

# Stack Together Store-Specific Covariates with General Covariates

store_covariates_future = []

for store in range(0, len(store_list)):
    stacked_covariates = holidays_transformed[store].stack(general_covariates)
    store_covariates_future.append(stacked_covariates)

store_covariates_past = []
holidays_transformed_sliced = holidays_transformed  # for slicing past covariates

for store in range(0, len(store_list)):
    holidays_transformed_sliced[store] = holidays_transformed[store].slice_intersect(transactions_covs[store])
    general_covariates_sliced = general_covariates.slice_intersect(transactions_covs[store])
    stacked_covariates = transactions_covs[store].stack(holidays_transformed_sliced[store]).stack(
        general_covariates_sliced)
    store_covariates_past.append(stacked_covariates)

# Save binary files
with open('vars/family_TS_dict.pkl', 'wb') as f:
    pickle.dump(family_TS_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/family_TS_transformed_dict.pkl', 'wb') as f:
    pickle.dump(family_TS_transformed_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/oil_transformed.pkl', 'wb') as f:
    pickle.dump(oil_transformed, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/sales_moving_averages_dict.pkl', 'wb') as f:
    pickle.dump(sales_moving_averages_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/store_covariates_future.pkl', 'wb') as f:
    pickle.dump(store_covariates_future, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/transactions_covs.pkl', 'wb') as f:
    pickle.dump(transactions_covs, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/time_cov_transformed.pkl', 'wb') as f:
    pickle.dump(time_cov_transformed, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/oil_moving_averages.pkl', 'wb') as f:
    pickle.dump(oil_moving_averages, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/list_of_holidays_per_store.pkl', 'wb') as f:
    pickle.dump(list_of_holidays_per_store, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/family_list.pkl', 'wb') as f:
    pickle.dump(family_list, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/transactions_transformed.pkl', 'wb') as f:
    pickle.dump(transactions_transformed, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/family_pipeline_dict.pkl', 'wb') as f:
    pickle.dump(family_pipeline_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vars/store_covariates_past.pkl', 'wb') as f:
    pickle.dump(store_covariates_past, f, protocol=pickle.HIGHEST_PROTOCOL)

# Delete Original Dataframes to Save Memory
del df_train
del df_stores
del df_holidays_events
del df_oil
del df_transactions
gc.collect()
