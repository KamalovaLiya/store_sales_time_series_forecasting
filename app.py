import streamlit as st

# Импортируем функции из ваших .py файлов
from upload import upload_data
from eda import perform_eda
from submission import get_submission


def main():
    st.title('Приложение для прогнозирования продаж в магазине')

    # Меню с вкладками для навигации
    menu = ["Загрузка", "Анализ данных", "Получение результатов"]
    choice = st.sidebar.selectbox("Меню", menu)

    if choice == "Загрузка":
        upload_data()

    elif choice == "Анализ данных":
        perform_eda()

    elif choice == "Получение результатов":
        get_submission()


if __name__ == "__main__":
    main()
