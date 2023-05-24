import streamlit as st
import forecast_analysis

from upload import upload_data
from eda import perform_eda
from submission import get_submission


def main():
    st.title('Приложение для прогнозирования продаж в магазине')

    # Добавляем новый пункт в меню для анализа прогнозов
    menu = ["Загрузка", "Анализ данных", "Получение результатов", "Анализ прогнозов"]
    choice = st.sidebar.selectbox("Меню", menu)

    if choice == "Загрузка":
        upload_data()

    elif choice == "Анализ данных":
        perform_eda()

    elif choice == "Получение результатов":
        get_submission()

    # Добавляем новую ветку для анализа прогнозов
    elif choice == "Анализ прогнозов":
        forecast_analysis.run()


if __name__ == "__main__":
    main()
