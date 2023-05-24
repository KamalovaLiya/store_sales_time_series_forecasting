import pandas as pd
import pickle
import numpy as np
import gc
import streamlit as st

from tqdm import tqdm
from darts.dataprocessing import Pipeline

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer, MissingValuesFiller, InvertibleMapper
from darts.models import MovingAverage


# @st.cache(allow_output_mutation=True)
def process_data_future_cov(promotion_transformed_dict, store_covariates_future):
    future_covariates_dict = {}
    for key in tqdm(promotion_transformed_dict):
        promotion_family = promotion_transformed_dict[key]
        covariates_future = [promotion_family[i].stack(store_covariates_future[i]) for i in
                             range(0, len(promotion_family))]
        future_covariates_dict[key] = covariates_future
    return future_covariates_dict


def process_data_promo(family_promotion_dict):
    promotion_transformed_dict = {}

    for key in tqdm(family_promotion_dict):
        promo_filler = MissingValuesFiller(verbose=False, n_jobs=-1, name="Fill NAs")
        promo_scaler = Scaler(verbose=False, n_jobs=-1, name="Scaling")

        promo_pipeline = Pipeline([promo_filler,
                                   promo_scaler])

        promotion_transformed = promo_pipeline.fit_transform(family_promotion_dict[key])

        # Moving Averages for Promotion Family Dictionaries
        promo_moving_average_7 = MovingAverage(window=7)
        promo_moving_average_28 = MovingAverage(window=28)

        promotion_covs = []

        for ts in promotion_transformed:
            ma_7 = promo_moving_average_7.filter(ts)
            ma_7 = TimeSeries.from_series(ma_7.pd_series())
            ma_7 = ma_7.astype(np.float32)
            ma_7 = ma_7.with_columns_renamed(col_names=ma_7.components, col_names_new="promotion_ma_7")
            ma_28 = promo_moving_average_28.filter(ts)
            ma_28 = TimeSeries.from_series(ma_28.pd_series())
            ma_28 = ma_28.astype(np.float32)
            ma_28 = ma_28.with_columns_renamed(col_names=ma_28.components, col_names_new="promotion_ma_28")
            promo_and_mas = ts.stack(ma_7).stack(ma_28)
            promotion_covs.append(promo_and_mas)
        promotion_transformed_dict[key] = promotion_covs
    return promotion_transformed_dict


def upload_data():
    df_train_parts = []
    for i in range(31):
        df_temp = pd.read_parquet(f'data/train/df_train_part_{i}.parquet')
        df_train_parts.append(df_temp)
    df_train = pd.concat(df_train_parts, ignore_index=True)
    with open('vars/family_list.pkl', 'rb') as f:
        family_list = pickle.load(f)

    with open('vars/transactions_covs.pkl', 'rb') as f:
        transactions_covs = pickle.load(f)
    with open('vars/sales_moving_averages_dict.pkl', 'rb') as f:
        sales_moving_averages_dict = pickle.load(f)

    with open('vars/store_covariates_future.pkl', 'rb') as f:
        store_covariates_future = pickle.load(f)
    with open('vars/store_covariates_past.pkl', 'rb') as f:
        store_covariates_past = pickle.load(f)

    st.header('Загрузка данных')
    file = st.file_uploader('Загрузите файл с данными', type=['csv'])
    if file is not None:
        df_test = pd.read_csv(file)
        df_test_dropped = df_test.drop(['onpromotion'], axis=1)
        df_test_sorted = df_test_dropped.sort_values(by=['store_nbr', 'family'])

        # Store/Family-Varying Covariates (Promotion)

        df_promotion = pd.concat([df_train, df_test], axis=0)
        df_promotion = df_promotion.sort_values(["store_nbr", "family", "date"])
        df_promotion.tail()

        family_promotion_dict = {}

        for family in family_list:
            df_family = df_promotion.loc[df_promotion['family'] == family]

            list_of_TS_promo = TimeSeries.from_group_dataframe(
                df_family,
                time_col="date",
                group_cols=["store_nbr", "family"],
                # individual time series are extracted by grouping `df` by `group_cols`
                value_cols="onpromotion",  # covariate of interest
                fill_missing_dates=True,
                freq='D')

            for ts in list_of_TS_promo:
                ts = ts.astype(np.float32)

            family_promotion_dict[family] = list_of_TS_promo

        # 2.5. Assemble All Covariates in Dictionaries
        # future_covariates_dict = process_data(promotion_transformed_dict, store_covariates_future)
        promotion_transformed_dict = process_data_promo(family_promotion_dict)
        past_covariates_dict = {}

        for key in tqdm(promotion_transformed_dict):
            promotion_family = promotion_transformed_dict[key]
            sales_mas = sales_moving_averages_dict[key]
            covariates_past = [promotion_family[i].slice_intersect(store_covariates_past[i]).stack(
                store_covariates_past[i].stack(sales_mas[i])) for i in range(0, len(promotion_family))]

            past_covariates_dict[key] = covariates_past

        only_past_covariates_dict = {}

        for key in tqdm(sales_moving_averages_dict):
            sales_moving_averages = sales_moving_averages_dict[key]
            only_past_covariates = [sales_moving_averages[i].stack(transactions_covs[i]) for i in
                                    range(0, len(sales_moving_averages))]

            only_past_covariates_dict[key] = only_past_covariates

        with open('vars/df_test_sorted.pkl', 'wb') as f:
            pickle.dump(df_test_sorted, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open('vars/family_promotion_dict.pkl', 'wb') as f:
            pickle.dump(family_promotion_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        del df_train
        del df_test
        gc.collect()
