from tqdm import tqdm
import pickle
import pandas as pd
import streamlit as st

from upload import process_data_future_cov, process_data_promo


# with open('vars/future_covariates_dict.pkl', 'rb') as f:
#     future_covariates_dict = pickle.load(f)

def get_submission():
    with open('model/lgbm_model.pkl', 'rb') as f:
        LGBM_Model_Submission = pickle.load(f)

    with open('vars/family_list.pkl', 'rb') as f:
        family_list = pickle.load(f)

    with open('vars/family_TS_transformed_dict.pkl', 'rb') as f:
        family_TS_transformed_dict = pickle.load(f)

    with open('vars/transactions_transformed.pkl', 'rb') as f:
        transactions_transformed = pickle.load(f)

    with open('vars/family_pipeline_dict.pkl', 'rb') as f:
        family_pipeline_dict = pickle.load(f)

    with open('vars/family_TS_dict.pkl', 'rb') as f:
        family_TS_dict = pickle.load(f)

    with open('vars/df_test_sorted.pkl', 'rb') as f:
        df_test_sorted = pickle.load(f)
    with open('vars/store_covariates_future.pkl', 'rb') as f:
        store_covariates_future = pickle.load(f)
    with open('vars/family_promotion_dict.pkl', 'rb') as f:
        family_promotion_dict = pickle.load(f)

    promotion_transformed_dict = process_data_promo(family_promotion_dict)

    future_covariates_dict = process_data_future_cov(promotion_transformed_dict, store_covariates_future)

    st.header('Получение результатов')
    # Generate Forecasts for Submission
    LGBM_Models_Submission = {}
    for family in tqdm(family_list):
        # Define Data for family
        sales_family = family_TS_transformed_dict[family]
        training_data = [ts for ts in sales_family]
        TCN_covariates = future_covariates_dict[family]
        train_sliced = [training_data[i].slice_intersect(TCN_covariates[i]) for i in range(0, len(training_data))]
        LGBM_Models_Submission[family] = LGBM_Model_Submission

    # Generate Forecasts for Submission

    LGBM_Forecasts_Families_Submission = {}

    for family in tqdm(family_list):
        sales_family = family_TS_transformed_dict[family]
        training_data = [ts for ts in sales_family]
        LGBM_covariates = future_covariates_dict[family]
        train_sliced = [training_data[i].slice_intersect(TCN_covariates[i]) for i in range(0, len(training_data))]

        forecast_LGBM = LGBM_Models_Submission[family].predict(n=16,
                                                               series=train_sliced,
                                                               future_covariates=LGBM_covariates,
                                                               past_covariates=transactions_transformed)

        LGBM_Forecasts_Families_Submission[family] = forecast_LGBM

    # Transform Back

    LGBM_Forecasts_Families_back_Submission = {}

    for family in tqdm(family_list):
        LGBM_Forecasts_Families_back_Submission[family] = family_pipeline_dict[family].inverse_transform(
            LGBM_Forecasts_Families_Submission[family], partial=True)

    # Zero Forecasting

    for family in tqdm(LGBM_Forecasts_Families_back_Submission):
        for n in range(0, len(LGBM_Forecasts_Families_back_Submission[family])):
            if (family_TS_dict[family][n].univariate_values()[-21:] == 0).all():
                LGBM_Forecasts_Families_back_Submission[family][n] = LGBM_Forecasts_Families_back_Submission[family][
                    n].map(
                    lambda x: x * 0)

    # Prepare Submission in Correct Format

    listofseries = []

    for store in range(0, 54):
        for family in tqdm(family_list):
            oneforecast = LGBM_Forecasts_Families_back_Submission[family][store].pd_dataframe()
            oneforecast.columns = ['fcast']
            listofseries.append(oneforecast)

    df_forecasts = pd.concat(listofseries)
    df_forecasts.reset_index(drop=True, inplace=True)

    # No Negative Forecasts
    df_forecasts[df_forecasts < 0] = 0
    forecasts_kaggle = pd.concat([df_test_sorted, df_forecasts.set_index(df_test_sorted.index)], axis=1)
    forecasts_kaggle_sorted = forecasts_kaggle.sort_values(by=['id'])
    forecasts_kaggle_sorted = forecasts_kaggle_sorted.drop(['date', 'store_nbr', 'family'], axis=1)
    forecasts_kaggle_sorted = forecasts_kaggle_sorted.rename(columns={"fcast": "sales"})
    forecasts_kaggle_sorted = forecasts_kaggle_sorted.reset_index(drop=True)

    # Submission
    submission_kaggle = forecasts_kaggle_sorted
    csv = submission_kaggle.to_csv(index=False)
    st.download_button(
        label="Download sales forecast CSV file",
        data=csv,
        file_name="submission_lgbm.csv",
        mime="text/csv"
    )
