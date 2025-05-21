import logging
from pathlib import Path
import time
from datetime import timedelta

from OCR_scripts.formula_recognizer import FormulaRecognizer


# Конфигурация
SOURCE_PATH = 'data/math_test/img6.jpg'  # Может быть путем или байтами
OUTPUT_DIR = 'data/cropped_lines'
IMG_FORMAT = 'png'


# Настройка логирования
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


def get_image_bytes() -> bytes:
    """Пример получения байтов изображения"""
    # Вариант 1: Из файла (для теста)
    with open(SOURCE_PATH, 'rb') as f:
        return f.read()

    # Вариант 2: Из API/сети
    # import requests
    # response = requests.get('http://example.com/image.jpg')
    # return response.content

    # Вариант 3: Из интерфейса (например, загрузка через Flask)
    # from flask import request
    # return request.files['image'].read()


def process_formulas(is_bytes=False):
    """Основная функция обработки"""
    start_time = time.time()
    setup_logging()

    try:
        recognizer = FormulaRecognizer()

        if is_bytes:
            # Вариант с байтами
            image_bytes = get_image_bytes()  # Функция для получения байтов
            line_paths = recognizer.segment_lines(
                image_bytes,
                OUTPUT_DIR,
                IMG_FORMAT,
                is_bytes=is_bytes
            )
        else:
            # Вариант с файлом
            source_path = Path(SOURCE_PATH)
            if not source_path.exists():
                logging.error(f'Path {source_path} does not exist')
                return

            line_paths = recognizer.segment_lines(
                source_path,
                OUTPUT_DIR,
                IMG_FORMAT,
                is_bytes=is_bytes
            )

        if not line_paths:
            logging.warning("Не удалось выделить строки")
            return None

        # Распознавание формул
        latex_outputs = []
        for line_path in line_paths:
            try:
                latex = recognizer.recognize_formula(line_path)
                latex_outputs.append(latex)
                logging.info(f"Распознано: {latex}")
            except Exception as e:
                logging.error(f"Ошибка распознавания {line_path}: {e}")

        # Объединение результатов
        combined_latex = " \\\\\n".join(latex_outputs)

        # Создание PDF
        try:
            recognizer.create_latex_pdf(combined_latex)
            logging.info("PDF успешно создан")
        except Exception as e:
            logging.error(f"Ошибка создания PDF: {e}")

        return combined_latex

    finally:
        # Очистка
        recognizer.cleanup(OUTPUT_DIR)
        elapsed_time = time.time() - start_time
        logging.info(f"Время выполнения: {timedelta(seconds=elapsed_time)}")


def main():
    # Вариант 1: Работа с файлом
    # result = process_formulas(is_bytes=False)

    # Вариант 2: Работа с байтами
    result = process_formulas(is_bytes=True)

    if result:
        print("\nИтоговый LaTeX код:")
        print(result)


if __name__ == "__main__":
    main()
