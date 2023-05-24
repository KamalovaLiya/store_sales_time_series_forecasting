import streamlit as st

import numpy as np
import pickle
import matplotlib.pyplot as plt
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis, plot_hist
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer, MissingValuesFiller, InvertibleMapper
from darts.dataprocessing import Pipeline
import sklearn
from sklearn import preprocessing
from upload import process_data_promo


def perform_eda():
    with open('vars/family_TS_dict.pkl', 'rb') as f:
        family_TS_dict = pickle.load(f)

    with open('vars/family_TS_transformed_dict.pkl', 'rb') as f:
        family_TS_transformed_dict = pickle.load(f)

    with open('vars/sales_moving_averages_dict.pkl', 'rb') as f:
        sales_moving_averages_dict = pickle.load(f)
    with open('vars/family_promotion_dict.pkl', 'rb') as f:
        family_promotion_dict = pickle.load(f)
    promotion_transformed_dict = process_data_promo(family_promotion_dict)

    with open('vars/transactions_covs.pkl', 'rb') as f:
        transactions_covs = pickle.load(f)

    with open('vars/oil_transformed.pkl', 'rb') as f:
        oil_transformed = pickle.load(f)

    with open('vars/oil_moving_averages.pkl', 'rb') as f:
        oil_moving_averages = pickle.load(f)

    with open('vars/time_cov_transformed.pkl', 'rb') as f:
        time_cov_transformed = pickle.load(f)

    with open('vars/list_of_holidays_per_store.pkl', 'rb') as f:
        list_of_holidays_per_store = pickle.load(f)
    st.header('Анализ данных')
    # Some EDA
    bread_series = family_TS_dict['BREAD/BAKERY'][0]
    celebration_series = family_TS_dict['CELEBRATION'][11]

    # Let's print two of the 1782 TimeSeries

    plt.subplots(2, 2, figsize=(15, 6))
    plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    bread_series.plot(label='Цены для {}'.format(bread_series.static_covariates_values()[0, 1],
                                                 bread_series.static_covariates_values()[0, 0],
                                                 bread_series.static_covariates_values()[0, 2]))

    celebration_series.plot(label='Цены для {}'.format(celebration_series.static_covariates_values()[0, 1],
                                                       celebration_series.static_covariates_values()[0, 0],
                                                       celebration_series.static_covariates_values()[0, 2]))

    plt.title(
        "Временные ряды для 2 семейств продукции для каждого из 2 магазинов (1-й и 12-й)")

    plt.subplot(1, 2, 2)  # index 2
    bread_series[-365:].plot(label='Ценя для {}'.format(bread_series.static_covariates_values()[0, 1],
                                                        bread_series.static_covariates_values()[0, 0],
                                                        bread_series.static_covariates_values()[0, 2]))

    celebration_series[-365:].plot(label='Цены для {}'.format(celebration_series.static_covariates_values()[0, 1],
                                                              celebration_series.static_covariates_values()[0, 0],
                                                              celebration_series.static_covariates_values()[0, 2]))

    plt.title("Только за последние 365 дней")
    st.pyplot(plt.gcf())  # Этот вызов заменяет plt.show()

    # Inspect Seasonality

    plot_acf(fill_missing_values(bread_series), m=7, alpha=0.05)
    plt.title("Автокорреляционная функция для {}, магазин {} в населенном пункте {}".format(
        bread_series.static_covariates_values()[0, 1],
        bread_series.static_covariates_values()[0, 0],
        bread_series.static_covariates_values()[0, 2]))
    st.pyplot(plt.gcf())  # Этот вызов заменяет plt.show()

    plot_acf(fill_missing_values(celebration_series), alpha=0.05)
    plt.title("Автокорреляционная функция для {}, магазин {} в в населенном пункте {}".format(
        celebration_series.static_covariates_values()[0, 1],
        celebration_series.static_covariates_values()[0, 0],
        celebration_series.static_covariates_values()[0, 2]));
    st.pyplot(plt.gcf())  # Этот вызов заменяет plt.show()

    # Show the Differenced Series

    # First Transform the Example Series
    train_filler_bread = MissingValuesFiller(verbose=False, n_jobs=-1, name="Fill NAs")
    static_cov_transformer_bread = StaticCovariatesTransformer(verbose=False,
                                                               transformer_cat=sklearn.preprocessing.OneHotEncoder(),
                                                               name="Encoder")
    log_transformer_bread = InvertibleMapper(np.log1p, np.expm1, verbose=False, n_jobs=-1, name="Log-Transform")
    train_scaler_bread = Scaler(verbose=False, n_jobs=-1, name="Scaling")

    train_filler_celebration = MissingValuesFiller(verbose=False, n_jobs=-1, name="Fill NAs")
    static_cov_transformer_celebration = StaticCovariatesTransformer(verbose=False,
                                                                     transformer_cat=sklearn.preprocessing.OneHotEncoder(),
                                                                     name="Encoder")
    log_transformer_celebration = InvertibleMapper(np.log1p, np.expm1, verbose=False, n_jobs=-1, name="Log-Transform")
    train_scaler_celebration = Scaler(verbose=False, n_jobs=-1, name="Scaling")

    train_pipeline_bread = Pipeline([train_filler_bread,
                                     static_cov_transformer_bread,
                                     log_transformer_bread,
                                     train_scaler_bread])

    train_pipeline_celebration = Pipeline([train_filler_celebration,
                                           static_cov_transformer_celebration,
                                           log_transformer_celebration,
                                           train_scaler_celebration])

    bread_series_transformed = train_pipeline_bread.fit_transform(bread_series)
    celebration_series_transformed = train_pipeline_celebration.fit_transform(celebration_series)

    # Plots

    plt.subplots(2, 2, figsize=(15, 6))
    plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    bread_series_transformed.plot(label='Цены для {}'.format(bread_series.static_covariates_values()[0, 1],
                                                              bread_series.static_covariates_values()[0, 0],
                                                              bread_series.static_covariates_values()[0, 2]))

    plt.title("Временные ряды после масштабирования и логарифмического преобразования")

    plt.subplot(1, 2, 2)  # index 2
    bread_series_transformed[-365:].plot(label='Ценя для {}'.format(bread_series.static_covariates_values()[0, 1],
                                                                     bread_series.static_covariates_values()[0, 0],
                                                                     bread_series.static_covariates_values()[0, 2]))

    plt.title("Только за последние 365 дней")
    st.pyplot(plt.gcf())

    plt.subplots(2, 2, figsize=(15, 6))
    plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    celebration_series_transformed.plot(label='Цены для {}'.format(celebration_series.static_covariates_values()[0, 1],
                                                                   celebration_series.static_covariates_values()[0, 0],
                                                                   celebration_series.static_covariates_values()[
                                                                       0, 2]))

    plt.title("Временные ряды после масштабирования и логарифмического преобразования")

    plt.subplot(1, 2, 2)  # index 2
    celebration_series_transformed[-365:].plot(
        label='Цены для {}'.format(celebration_series.static_covariates_values()[0, 1],
                                   celebration_series.static_covariates_values()[0, 0],
                                   celebration_series.static_covariates_values()[0, 2]))

    plt.title("Только за последние 365 дней")
    st.pyplot(plt.gcf())

    plt.figure(figsize=(10, 6))
    family_TS_transformed_dict['BREAD/BAKERY'][0][-180:].plot()
    sales_moving_averages_dict['BREAD/BAKERY'][0][-180:].plot()
    plt.title("Продажи 7- и 28-дневные скользящие средние")
    st.pyplot(plt.gcf());

    plt.figure(figsize=(10, 6))
    promotion_transformed_dict['BREAD/BAKERY'][0][-180:].plot()
    plt.title("Данные о продвижении и скользящие средние")
    st.pyplot(plt.gcf());

    plt.figure(figsize=(10, 6))
    transactions_covs[0][-180:].plot()
    plt.legend(loc='lower right')
    plt.title("Данные о транзакциях и скользящие средние")
    st.pyplot(plt.gcf());

    plt.figure(figsize=(10, 6))
    oil_transformed[-180:].plot()
    oil_moving_averages[-180:].plot()
    plt.title("Цена на нефть и скользящие средние")
    st.pyplot(plt.gcf());

    # plt.figure(figsize=(10, 6))
    # time_cov_transformed[-180:].plot()
    # plt.title("Ковариаты, связанные со временем")
    # st.pyplot(plt.gcf());

    # df_holidays_events['type'].value_counts().plot.bar(rot=0)
    plt.figure(figsize=(10, 6))
    list_of_holidays_per_store[0].loc[:, list_of_holidays_per_store[0].columns != "date"].sum().plot.bar(rot=0)
    plt.title("Праздники и события")
    st.pyplot(plt.gcf());
