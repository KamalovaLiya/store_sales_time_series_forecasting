import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


def run():
    st.title('Анализ прогнозов продаж')

    # Загрузите данные прогнозов
    df = pd.read_csv('forecasts.csv')

    # Преобразуйте колонку date из строкового формата в формат даты
    df['date'] = pd.to_datetime(df['date'])

    # Добавьте возможность выбора группировки
    group_choice = st.selectbox(
        'Выберите способ группировки данных:',
        ('Дата', 'Номер магазина', 'Семейство продуктов',
         'Дата и номер магазина', 'Дата и семейство продуктов',
         'Номер магазина и семейство продуктов', 'Дата, номер магазина и семейство продуктов')
    )
    col_map = {'Номер магазина': 'store_nbr', 'Семейство продуктов': 'family', 'Дата': 'date'}

    if group_choice == 'Дата':
        date = st.date_input('Выберите дату')
        axis_choice = st.selectbox(
            'Выберите, что будет на оси X:',
            ('Номер магазина', 'Семейство продуктов')
        )
        subset = df[df['date'] == pd.Timestamp(date)]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(subset[col_map[axis_choice]], subset['sales'])
        plt.xticks(rotation=90)
        plt.title('Продажи по ' + axis_choice)
        plt.xlabel(axis_choice)
        plt.ylabel('Продажи')
        st.pyplot(fig)

    elif group_choice == 'Номер магазина':
        store = st.selectbox('Выберите номер магазина', df['store_nbr'].unique())
        axis_choice = st.selectbox(
            'Выберите, что будет на оси X:',
            ('Дата', 'Семейство продуктов')
        )
        subset = df[df['store_nbr'] == store]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(subset[col_map[axis_choice]], subset['sales'])
        if axis_choice == 'Дата':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
            fig.autofmt_xdate()
        else:
            plt.xticks(rotation=90)
        plt.title('Продажи по ' + axis_choice)
        plt.xlabel(axis_choice)
        plt.ylabel('Продажи')
        st.pyplot(fig)

    elif group_choice == 'Семейство продуктов':
        family = st.selectbox('Выберите семейство продуктов', df['family'].unique())
        axis_choice = st.selectbox(
            'Выберите, что будет на оси X:',
            ('Дата', 'Номер магазина')
        )
        subset = df[df['family'] == family]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(subset[col_map[axis_choice]], subset['sales'])
        if axis_choice == 'Дата':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
            fig.autofmt_xdate()
        else:
            plt.xticks(rotation=90)
        plt.title('Продажи по ' + axis_choice)
        plt.xlabel(axis_choice)
        plt.ylabel('Продажи')
        st.pyplot(fig)

    elif group_choice == 'Дата и номер магазина':
        store = st.selectbox('Выберите номер магазина', df['store_nbr'].unique())
        family = st.selectbox('Выберите семейство продуктов', df['family'].unique())
        subset = df[(df['store_nbr'] == store) & (df['family'] == family)]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(subset['date'], subset['sales'])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
        fig.autofmt_xdate()
        plt.title('Продажи по времени')
        plt.xlabel('Дата')
        plt.ylabel('Продажи')
        st.pyplot(fig)

    elif group_choice == 'Дата и семейство продуктов':
        date = st.date_input('Выберите дату')
        family = st.selectbox('Выберите семейство продуктов', df['family'].unique())
        subset = df[(df['date'] == pd.Timestamp(date)) & (df['family'] == family)]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(subset['store_nbr'], subset['sales'])
        plt.xticks(rotation=90)
        plt.title('Продажи по номеру магазина')
        plt.xlabel('Номер магазина')
        plt.ylabel('Продажи')
        st.pyplot(fig)

    elif group_choice == 'Номер магазина и семейство продуктов':
        store = st.selectbox('Выберите номер магазина', df['store_nbr'].unique())
        date = st.date_input('Выберите дату')
        subset = df[(df['date'] == pd.Timestamp(date)) & (df['store_nbr'] == store)]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(subset['family'], subset['sales'])
        plt.xticks(rotation=90)
        plt.title('Продажи по семейству продуктов')
        plt.xlabel('Семейство продуктов')
        plt.ylabel('Продажи')
        st.pyplot(fig)

    elif group_choice == 'Дата, номер магазина и семейство продуктов':
        date = st.date_input('Выберите дату')
        store = st.selectbox('Выберите номер магазина', df['store_nbr'].unique())
        family = st.selectbox('Выберите семейство продуктов', df['family'].unique())
        subset = df[(df['date'] == pd.Timestamp(date)) & (df['store_nbr'] == store) & (df['family'] == family)]
        if len(subset) > 0:
            st.write(f"Продажи: {subset['sales'].values[0]}")
        else:
            st.write("Продажи: Не найдено")
