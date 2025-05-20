import time
from datetime import timedelta

from pathlib import Path

import logging

from OCR_scripts.segmentation import ImageSegmenter
from OCR_scripts.recognition import OCRPredictor
from OCR_scripts.config import Config


# ========== НАСТРОЙКИ СЕГМЕНТАЦИИ ==========
MODEL_PATH = 'models/model.pt'  # Модель сегментации строк
SOURCE_PATH = 'data/my_test/img22.jpg'  # Путь до изображения
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


def get_bboxes():
    # Логирование
    setup_logging()

    segmenter = ImageSegmenter(
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        overlap_threshold=OVERLAP_THRESHOLD,
        scale_coeff=SCALE_COEFF,
        scale_bbox=SCALE_BBOX,
        space_threshold_coeff=SPACE_THRESHOLD_COEFF,
        min_space_width=MIN_SPACE_WIDTH
    )

    # Создаем выходную директорию
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Обрабатываем исходный путь
    source_path = Path(SOURCE_PATH)
    if not source_path.exists():
        logging.error(f'Path {source_path} does not exist')
        return

    if source_path.is_file():
        segmenter.process_image(source_path, output_dir, IMG_FORMAT)
    elif source_path.is_dir():
        for img_path in source_path.glob('*.*'):
            if img_path.suffix.lower()[1:] in ['jpg', 'jpeg', 'png']:
                segmenter.process_image(img_path, output_dir, IMG_FORMAT)


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
    predictor.print_results(predictions)


def main():
    # Начало отсчёта времени
    start_time = time.time()

    # Получение ббоксов
    get_bboxes()

    # Распознавание текста
    recognition_text()

    # Конец отсчёта времени
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(timedelta(seconds=elapsed_time))
    print(f"\nПрограмма завершена. Общее время выполнения: {elapsed_time_str}")


if __name__ == '__main__':
    main()
