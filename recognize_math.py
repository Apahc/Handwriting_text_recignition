from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import subprocess
import os
import cv2
import numpy as np
import logging
from pathlib import Path
import shutil

# ========== НАСТROYKI ==========
SOURCE_PATH = 'data/math_test/img6.jpg'                 # Путь к изображению с формулами
OUTPUT_DIR = 'data/cropped_lines'                       # Папка для сохранения вырезанных строк
IMG_FORMAT = 'png'                                      # Формат изображений
# Параметры для сегментации без YOLO
THRESHOLD = 200                                         # Порог бинаризации
MIN_LINE_HEIGHT = 10                                    # Минимальная высота строки (пиксели)
MIN_GAP_HEIGHT = 5                                      # Минимальный зазор между строками (пиксели)
# ================================

# Настройка логирования
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('processing.log'), logging.StreamHandler()]
    )

# Загрузка процессора и модели TrOCR
try:
    processor = TrOCRProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten", use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")
except Exception as e:
    print(f"Ошибка загрузки модели TrOCR: {e}")
    exit()

# Установка устройства (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Альтернативная сегментация строк (без YOLO)
def segment_lines_without_yolo(image_path):
    setup_logging()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(image_path)
    if not source_path.exists():
        logging.error(f'Path {source_path} does not exist')
        return []

    # Загрузка и обработка изображения
    img = cv2.imread(str(source_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Улучшение контраста
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
    # Бинаризация
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
    # Добавление последней строки, если она есть
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
                logging.warning(f"Empty crop for line {i} in {image_path}")
                continue

            output_name = f"{source_filename}_line_{i}.{IMG_FORMAT}"
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), cropped_img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100] if IMG_FORMAT == 'jpg' else [])
            logging.info(f"Saved line {i} to {output_path}")
            line_paths.append(str(output_path))

            # Аннотация
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as e:
            logging.error(f'Error processing line {i} in {image_path}: {str(e)}')

    annotated_path = output_dir / f"{source_filename}_annotated.{IMG_FORMAT}"
    cv2.imwrite(str(annotated_path), annotated_img)
    logging.info(f'Saved annotated image: {annotated_path}')

    return line_paths

def recognize_formula(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    generated_ids = model.generate(
        pixel_values,
        max_length=100,
        num_beams=10,
        early_stopping=True,
        temperature=0.7
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def create_latex_pdf(latex_code, output_file="formula.pdf"):
    latex_template = r"""
    \documentclass{article}
    \usepackage{amsmath}
    \begin{document}
    \begin{align*}
    %s
    \end{align*}
    \end{document}
    """
    with open("temp.tex", "w", encoding="utf-8") as f:
        f.write(latex_template % latex_code)
    subprocess.run(["pdflatex", "temp.tex"])
    os.rename("temp.pdf", output_file)

def remove_output_directory(directory_path):
    """Удаляет указанную директорию и все её содержимое."""
    directory = Path(directory_path)
    if directory.exists() and directory.is_dir():
        try:
            shutil.rmtree(directory)
            logging.info(f'Папка {directory} успешно удалена.')
        except Exception as e:
            logging.error(f'Ошибка при удалении папки {directory}: {str(e)}')
    else:
        logging.warning(f'Папка {directory} не существует или не является директорией.')

def main():
    image_path = SOURCE_PATH
    output_file = "output.tex"
    pdf_file = "formula.pdf"

    # Сегментация строк без YOLO
    line_image_paths = segment_lines_without_yolo(image_path)
    if not line_image_paths:
        print("Не удалось выделить строки. Проверьте изображение и параметры сегментации.")
        print("Аннотированное изображение сохранено в data\cropped_lines\img6_annotated.png для анализа.")
        return

    # Распознавание каждой строки
    latex_outputs = []
    for line_path in line_image_paths:
        try:
            latex_output = recognize_formula(line_path)
            latex_outputs.append(latex_output)
            print(f"LaTeX output for {line_path}: {latex_output}")
        except Exception as e:
            print(f"Ошибка обработки строки {line_path}: {e}")
            continue

    # Объединяем LaTeX-код для всех строк
    combined_latex = " \\\\\n".join(latex_outputs)
    print("\nОбщий LaTeX-код:")
    print(combined_latex)

    # Сохранение в файл
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_latex)

    # Создание PDF
    try:
        create_latex_pdf(combined_latex, pdf_file)
        print(f"PDF сохранён как {pdf_file}")
    except FileNotFoundError:
        print("pdflatex не найден. Установите TeXLive или MiKTeX для создания PDF.")
    except Exception as e:
        print(f"Ошибка создания PDF: {e}")

    # Удаление временных файлов
    for temp_file in ["temp.tex", "temp.aux", "temp.log"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Удаление папки cropped_lines
    remove_output_directory(OUTPUT_DIR)

if __name__ == "__main__":
    main()