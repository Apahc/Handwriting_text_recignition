from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import numpy as np
from pathlib import Path
import shutil

# ========== НАСТРОЙКИ ==========
SOURCE_PATH = 'data/english_test/img1.jpg'  # Путь к изображению
OUTPUT_DIR = 'data/cropped_lines'         # Папка для сохранения вырезанных строк
IMG_FORMAT = 'png'                      # Формат изображений
THRESHOLD = 200                        # Порог бинаризации
MIN_LINE_HEIGHT = 30                   # Минимальная высота строки (пиксели)
MIN_GAP_HEIGHT = 5                     # Минимальный зазор между строками (пиксели)

# Загрузка процессора и модели TrOCR
local_path = "models/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(local_path, use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained(local_path)

# Функция сегментации строк
def segment_lines(image_path):
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(image_path)
    if not source_path.exists():
        print(f'Путь {source_path} не существует')
        return []

    # Загрузка и обработка изображения
    img = cv2.imread(str(source_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Горизонтальная проекция
    projection = np.sum(binary, axis=1) / 255

    # Поиск строк
    height = img.shape[0]
    line_bboxes = []
    start_y = None
    for y in range(height):
        if projection[y] > 0 and start_y is None:
            start_y = y
        elif projection[y] == 0 and start_y is not None:
            if y - start_y >= MIN_LINE_HEIGHT:
                line_bboxes.append((0, start_y, img.shape[1], y))
            start_y = None
    if start_y is not None and height - start_y >= MIN_LINE_HEIGHT:
        line_bboxes.append((0, start_y, img.shape[1], height))

    # Фильтрация строк по минимальному зазору
    filtered_bboxes = []
    prev_y2 = -MIN_GAP_HEIGHT - 1
    for (x1, y1, x2, y2) in line_bboxes:
        if y1 - prev_y2 >= MIN_GAP_HEIGHT:
            filtered_bboxes.append((x1, y1, x2, y2))
        prev_y2 = y2

    # Сохранение строк
    line_paths = []
    annotated_img = img.copy()
    source_filename = source_path.stem
    for i, (x1, y1, x2, y2) in enumerate(filtered_bboxes):
        try:
            cropped_img = img[y1:y2, x1:x2]
            if cropped_img.size == 0:
                print(f"Пустая строка {i} в {image_path}")
                continue

            output_name = f"{source_filename}_line_{i}.{IMG_FORMAT}"
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), cropped_img)
            print(f"Сохранена строка {i} в {output_path}")
            line_paths.append(str(output_path))

            # Аннотация
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as e:
            print(f'Ошибка обработки строки {i} в {image_path}: {str(e)}')

    annotated_path = output_dir / f"{source_filename}_annotated.{IMG_FORMAT}"
    cv2.imwrite(str(annotated_path), annotated_img)
    print(f'Сохранено аннотированное изображение: {annotated_path}')

    return line_paths

# Функция распознавания текста
def recognize_text(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Функция удаления директории
def remove_output_directory(directory_path):
    directory = Path(directory_path)
    if directory.exists() and directory.is_dir():
        try:
            shutil.rmtree(directory)
            print(f'Папка {directory} успешно удалена.')
        except Exception as e:
            print(f'Ошибка при удалении папки {directory}: {str(e)}')
    else:
        print(f'Папка {directory} не существует или не является директорией.')

# Основная функция
def main():
    image_path = SOURCE_PATH

    # Сегментация строк
    line_image_paths = segment_lines(image_path)
    if not line_image_paths:
        print("Не удалось выделить строки. Проверьте изображение и параметры сегментации.")
        print(f"Аннотированное изображение сохранено в {OUTPUT_DIR}/{Path(image_path).stem}_annotated.{IMG_FORMAT} для анализа.")
        return

    # Распознавание каждой строки
    recognized_texts = []
    for line_path in line_image_paths:
        try:
            text = recognize_text(line_path)
            recognized_texts.append(text)
            print(f"Текст для строки {line_path}: {text}")
        except Exception as e:
            print(f"Ошибка обработки строки {line_path}: {e}")
            continue

    # Вывод объединённого текста
    combined_text = "\n".join(recognized_texts)
    print("\nОбщий распознанный текст:")
    print(combined_text)

    # Удаление папки cropped_lines
    remove_output_directory(OUTPUT_DIR)

if __name__ == "__main__":
    main()