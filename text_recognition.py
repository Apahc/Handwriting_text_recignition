import time
from datetime import timedelta

from pathlib import Path

import logging

from OCR_scripts.segmentation import ImageSegmenter
from OCR_scripts.recognition import OCRPredictor
from OCR_scripts.config import Config
from OCR_scripts.answer_grader import AnswerGrader


# ========== НАСТРОЙКИ СЕГМЕНТАЦИИ ==========
MODEL_PATH = 'models/model.pt'                          # Модель сегментации строк
SOURCE_PATH = 'data/my_test/img22.jpg'                  # Путь до изображения
OUTPUT_DIR = 'data/cropped_boxes'                       # Папка куда сохраняются ббоксы
CONF_THRESHOLD = 0.3                                    # Порог уверенности (0-1)
OVERLAP_THRESHOLD = 0.35                                # Порог пересечения (50% площади каждого бокса)
IMG_FORMAT = 'png'                                      # Формат изображений (png/jpg)
SCALE_COEFF = 2                                         # Коэффициент растяжения
SCALE_BBOX = 0.01                                       # Процент увеличения ббокса
SPACE_THRESHOLD_COEFF = 0.0025                          # Процент принятия пробела
MIN_SPACE_WIDTH = 0.02                                  # Минимальная длинна пробела
# ==========================================


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


def get_image_bytes() -> bytes:
    """Пример получения байтов изображения"""
    # Вариант 1: Из файла (для теста)
    with open('data/my_test/img22.jpg', 'rb') as f:
        return f.read()

    # Вариант 2: Из API/сети
    # import requests
    # response = requests.get('http://example.com/image.jpg')
    # return response.content

    # Вариант 3: Из интерфейса (например, загрузка через Flask)
    # from flask import request
    # return request.files['image'].read()


def get_bboxes(is_bytes: bool = False):
    # Логирование
    setup_logging()

    segmenter = ImageSegmenter(MODEL_PATH)

    # Создаем выходную директорию
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Обрабатываем исходный путь
    if is_bytes:
        # Вариант с байтами
        image_bytes = get_image_bytes()     # Функция для получения байтов
        segmenter.process_image(image_bytes, output_dir, IMG_FORMAT, is_bytes)
    else:
        # Вариант с файлом
        source_path = Path(SOURCE_PATH)
        if not source_path.exists():
            logging.error(f'Path {source_path} does not exist')
            return

        segmenter.process_image(source_path, output_dir, IMG_FORMAT, is_bytes)


def recognition_text():
    # Конфигурация
    config = Config()
    recognition_model_path = "models/best_model-0.6780.pt"
    output_file = "predictions.json"

    # Инициализация предсказателя
    predictor = OCRPredictor(recognition_model_path, config)
    predictions = predictor.predict_from_folder(OUTPUT_DIR)

    # Сохранение и вывод результатов
    predictor.save_results_to_json(predictions, output_file)
    return predictor.get_and_print_results(predictions)


def evaluate_answers(student_answer: str) -> dict:
    grader = AnswerGrader()

    reference_answer = grader.get_reference_answer()

    is_correct, match_type = grader.grade_answer(student_answer, reference_answer)
    answer = {
        "is_correct": is_correct,
        "match_type": match_type,
        "student_answer": student_answer,
        "reference_answer": reference_answer
    }

    # Вывод результатов
    print("\nРезультаты оценки:")
    print(f"\nОтвет студента: {answer['student_answer']}")
    print(f"\nЭталонный ответ: {answer['reference_answer']}")
    print(f"\nРезультат: {'Правильно' if answer['is_correct'] else 'Неправильно'}")
    print(f"\nТип сравнения: {answer['match_type']}")

    return answer


def main():
    # Начало отсчёта времени
    start_time = time.time()

    # 1. Получение ббоксов
    # Вариант 1: Загрузка из файла
    # get_bboxes(is_bytes=False)

    # Вариант 2: Работа с байтами
    get_bboxes(is_bytes=True)

    # 2. Распознавание текста
    res_text = recognition_text()

    # 3. Оценка ответа
    evaluate_answers(res_text)

    # Конец отсчёта времени
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(timedelta(seconds=elapsed_time))
    print(f"\nПрограмма завершена. Общее время выполнения: {elapsed_time_str}")


if __name__ == '__main__':
    main()
