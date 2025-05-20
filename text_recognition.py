from ultralytics import YOLO
import cv2
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
from typing import List, Union
import requests
import time
from datetime import timedelta
import json


# ========== НАСТРОЙКИ ==========
MODEL_PATH = 'models/model.pt'
SOURCE_PATH = 'data/my_test/img24.jpg'
OUTPUT_DIR = 'data/cropped_boxes'
CONF_THRESHOLD = 0.3                                    # Порог уверенности (0-1)
OVERLAP_THRESHOLD = 0.9                                # Порог пересечения (50% площади каждого бокса)
IMG_FORMAT = 'png'                                      # Формат изображений (png/jpg)
SCALE_COEFF =2                                         # Коэффициент растяжения
SCALE_BBOX = 0.1                                       # Процент увеличения ббокса
SPACE_THRESHOLD_COEFF = 0.0001
MIN_SPACE_WIDTH = 0.015
# ================================


# Конфигурация (адаптируйте под ваши параметры)
class Config:
    def __init__(self):
        self.alphabet = " абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        self.image_width = 256
        self.image_height = 32
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Токенизатор (как в вашем исходном коде)
OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'


class Tokenizer:
    def __init__(self, alphabet):
        print("Initializing tokenizer...")
        self.char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
        self.char_map[CTC_BLANK] = 0
        self.char_map[OOV_TOKEN] = 1
        self.rev_char_map = {val: key for key, val in self.char_map.items()}
        print(f"Tokenizer initialized with {len(self.char_map)} characters")

    def decode(self, enc_word_list):
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                if (char_enc != self.char_map[OOV_TOKEN] and
                        char_enc != self.char_map[CTC_BLANK] and
                        not (idx > 0 and char_enc == word[idx - 1])):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words

    def get_num_chars(self):
        return len(self.char_map)


# Трансформы для изображений
class ImageResize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)


class Normalize:
    def __call__(self, img):
        return img.astype(np.float32) / 255


class ToTensor:
    def __call__(self, arr):
        return torch.from_numpy(arr)


class MoveChannels:
    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        return np.moveaxis(image, 0, -1)


def get_val_transforms(height, width):
    return torchvision.transforms.Compose([
        ImageResize(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])


# Модель CRNN (как в вашем исходном коде)
def get_resnet34_backbone(pretrained=True):
    m = torchvision.models.resnet34(pretrained=pretrained)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class CRNN(nn.Module):
    def __init__(self, number_class_symbols, time_feature_count=256, lstm_hidden=256, lstm_len=2):
        super().__init__()
        self.feature_extractor = get_resnet34_backbone(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x


# Класс для предсказания
class OcrPredictor:
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.tokenizer = Tokenizer(config.alphabet)
        self.device = torch.device(config.device)

        # Загружаем модель
        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transforms = get_val_transforms(config.image_height, config.image_width)
        print(f"Model loaded from {model_path} and ready for predictions")

    def predict_batch(self, images: List[np.ndarray]) -> List[str]:
        """Предсказание для батча изображений"""
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        images_tensor = torch.stack(transformed_images, 0).to(self.device)

        with torch.no_grad():
            output = self.model(images_tensor)

        pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
        return self.tokenizer.decode(pred)

    def predict_single(self, image: np.ndarray) -> str:
        """Предсказание для одного изображения"""
        return self.predict_batch([image])[0]

    def predict_from_folder(self, folder_path: str, batch_size: int = 32) -> dict:
        """Предсказание для всех изображений в папке"""
        results = {}
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)

        print(f"Found {total_images} images in {folder_path}")

        # Объединяем текст для обработки батчем (оптимизация запросов к Yandex Speller)
        all_predictions = []
        all_files = []

        # Обрабатываем батчами
        for i in range(0, total_images, batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_images = []

            for img_file in batch_files:
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    batch_images.append(img)
                else:
                    print(f"Warning: Could not read image {img_file}")

            if batch_images:
                predictions = self.predict_batch(batch_images)
                all_predictions.extend(predictions)
                all_files.extend(batch_files)

        # Корректировка текста батчем (если возможно, но Yandex Speller принимает только один текст за раз)
        if all_predictions:
            corrected_texts = [correct_text_with_yandex_speller(pred) for pred in all_predictions]
            print(all_predictions)
            print(corrected_texts)
            for img_file, corrected_pred in zip(all_files, corrected_texts):
                results[img_file] = corrected_pred

            print(f"Processed {total_images}/{total_images} images")

        return results


# Функция для корректировки текста с помощью Yandex Speller
def correct_text_with_yandex_speller(text):
    url = "https://speller.yandex.net/services/spellservice.json/checkText"
    params = {
        "text": text,
        "lang": "ru",
        "options": 6,
    }
    response = requests.get(url, params=params)
    corrections = response.json()

    corrected_text = text
    for correction in corrections:
        if correction.get("s"):
            wrong = correction["word"]
            right = correction["s"][0]
            corrected_text = corrected_text.replace(wrong, right)

    return corrected_text

def setup_logging():
    """Функция для логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('../../processing.log'), logging.StreamHandler()]
    )


def preprocess_image(image):
    """Растягиваем картинку в оттенках серого по Y"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray, None, fx=1, fy=SCALE_COEFF, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)


def calculate_intersection(box_a, box_b):
    """Вычисляет площадь пересечения двух bounding boxes"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def filter_boxes(boxes):
    """Фильтрует дублирующиеся bounding boxes """
    filtered = []
    for current_box in sorted(boxes, key=lambda x: x.conf, reverse=True):
        current_coords = current_box.xyxy[0].int().tolist()
        current_area = (current_coords[2] - current_coords[0]) * (current_coords[3] - current_coords[1])

        keep = True
        for idx, kept_box in enumerate(filtered):
            kept_coords = kept_box.xyxy[0].int().tolist()
            kept_area = (kept_coords[2] - kept_coords[0]) * (kept_coords[3] - kept_coords[1])

            # Вычисляем площадь пересечения
            inter_area = calculate_intersection(current_coords, kept_coords)

            # Проверяем условие пересечения
            if inter_area > OVERLAP_THRESHOLD * current_area and inter_area > OVERLAP_THRESHOLD * kept_area:
                # # Выбор большего бокса
                # if current_area > kept_area:
                #     filtered[idx] = current_box  # Замена на больший бокс
                keep = False
                break

        if keep:
            filtered.append(current_box)
    return filtered


def get_word_bboxes(img, x1, y1, x2, y2):
    # Обрезаем область строки
    crop = img[y1:y2, x1:x2]

    # Преобразуем в градации серого и бинаризуем
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Применяем морфологическое закрытие для объединения символов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Считаем горизонтальную проекцию
    projection = np.sum(closed, axis=0)
    # projection = cv2.medianBlur(projection.astype(np.uint8), 3)

    # Находим пробелы (адаптивный порог: % от максимальной высоты строки)
    height = y2 - y1
    space_threshold = SPACE_THRESHOLD_COEFF * height * 255  # % от максимально возможной суммы
    # median_projection = np.mean(projection[projection > 0])
    # space_threshold = 0.3 * median_projection  # 10% от медианной интенсивности символов

    # plt.plot(projection)
    # plt.axhline(y=space_threshold, color='r')
    # plt.show()

    space_indices = np.where(projection < space_threshold)[0]

    # Определяем группы непрерывных пробелов
    width = x2 - x1
    spaces = []
    if len(space_indices) > 0:
        start = space_indices[0]
        end = start
        for idx in space_indices[1:]:
            if idx - end == 1:
                end = idx
            else:
                # Проверяем ширину пробела перед добавлением
                if (end - start + 1) >= width * MIN_SPACE_WIDTH:
                    spaces.append((start, end))
                start = end = idx
        if (end - start + 1) >= width * MIN_SPACE_WIDTH:
            spaces.append((start, end))

    # Вычисляем границы слов
    word_boxes = []
    prev_end = 0
    for (s_start, s_end) in spaces:
        if s_start > prev_end:
            word_boxes.append((prev_end, 0, s_start, closed.shape[0]))
        prev_end = s_end

    # Добавляем последнее слово
    if prev_end < closed.shape[1]:
        word_boxes.append((prev_end, 0, closed.shape[1], closed.shape[0]))

    # Фильтруем мелкие артефакты и преобразуем в глобальные координаты
    width = x2 - x1
    min_word_width = width * 0.01  # Минимальная ширина слова в пикселях
    filtered = []
    for (wx1, wy1, wx2, wy2) in word_boxes:
        if wx2 - wx1 >= min_word_width:
            global_x1 = int((x1 + wx1) * 0.99)
            global_y1 = int((y1 + wy1) * 0.99)
            global_x2 = int((x1 + wx2) * 1.01)
            global_y2 = int((y1 + wy2) * 1.01)
            filtered.append((global_x1, global_y1, global_x2, global_y2))

            cv2.rectangle(img, (global_x1, global_y1), (global_x2, global_y2), (0, 255, 0), 2)

    return filtered


def sort_word_bboxes(word_bboxes, line_overlap_threshold=0.7):
    """
    Сортирует bounding boxes по правилам:
    1. По вертикали (по y1)
    2. Для объектов в одной строке - по горизонтали (по x1)

    :param word_bboxes: список кортежей (x1, y1, x2, y2)
    :param line_overlap_threshold: порог перекрытия для определения одной строки (0.5 = 50%)
    :return: отсортированный список координат
    """
    # Рассчитываем высоты bounding boxes
    bboxes_with_height = [(box, box[3] - box[1]) for box in word_bboxes]

    # Сортируем по вертикали (основной критерий - y1)
    sorted_boxes = sorted(bboxes_with_height, key=lambda x: x[0][1])

    # Группируем по строкам
    lines = []
    for box, h in sorted_boxes:
        y_center = (box[1] + box[3]) / 2  # Центр по вертикали
        matched = False

        # Проверяем существующие линии
        for line in lines:
            # Берем эталонный bbox из линии
            ref_box = line[0][0]
            line_height = ref_box[3] - ref_box[1]

            # Порог перекрытия с учетом высоты линии
            threshold = line_height * line_overlap_threshold

            # Проверяем вертикальное перекрытие
            if ref_box[1] - threshold <= y_center <= ref_box[3] + threshold:
                line.append((box, h))
                matched = True
                break

        # Если не нашли подходящую линию, создаем новую
        if not matched:
            lines.append([(box, h)])

    # Сортируем каждую линию по x1 и формируем итоговый список
    result = []
    for line in lines:
        # Сортировка внутри линии
        line_sorted = sorted(line, key=lambda x: x[0][0])
        # Оставляем только координаты
        result.extend([box for box, h in line_sorted])

    return result


def main0():
    # Логирование
    setup_logging()

    try:
        model = YOLO(MODEL_PATH)
        logging.info(f'Model "{MODEL_PATH}" loaded')
    except Exception as e:
        logging.error(f'Error loading model: {str(e)}')
        return

    # Создаем выходную директорию
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Обрабатываем исходный путь
    source_path = Path(SOURCE_PATH)
    if not source_path.exists():
        logging.error(f'Path {source_path} does not exist')
        return

    image_paths = []
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        image_paths = list(source_path.glob('*.*'))

    for img_path in image_paths:
        if img_path.suffix.lower()[1:] not in ['jpg', 'jpeg', 'png']:
            continue

        logging.info(f'Processing: {img_path}')
        original_img = cv2.imread(str(img_path))
        processed_img = preprocess_image(original_img)
        plt.show()

        results = model.predict(source=processed_img, iou=0.2, agnostic_nms=True)

        for result in results:
            # Получаем исходное изображение
            annotated_img = original_img.copy()
            source_filename = img_path.stem

            # Фильтрация боксов
            boxes = [box for box in result.boxes if box.conf >= CONF_THRESHOLD]
            filtered_boxes = filter_boxes(boxes)

            # Сортировка боксов: сверху-вниз, слева-направо
            sorted_boxes = sorted(filtered_boxes,
                                  key=lambda b: (
                                      b.xyxy[0][1].item(),  # Сортировка по Y (верхняя граница)
                                      b.xyxy[0][0].item()  # Затем по X (левая граница)
                                  ))

            # Получаем информацию о bounding boxes
            for i, box in enumerate(sorted_boxes):
                if box.conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())     # Координаты bbox в формате (x1, y1, x2, y2)
                class_id = int(box.cls[0].item())                   # Confidence score
                class_name = model.names.get(class_id, 'unknown')   # Class ID (индекс класса)
                y1 //= SCALE_COEFF                                   # Возвращение к исходному формату y1
                y2 //= SCALE_COEFF                                   # Возвращение к исходному формату y2

                word_bboxes = get_word_bboxes(annotated_img, x1, y1, x2, y2)
                word_bboxes = sort_word_bboxes(word_bboxes)

                for j, (x1, y1, x2, y2) in enumerate(word_bboxes):
                    # Вырезаем изображение
                    try:
                        # Проверка координат
                        h, w = original_img.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x1 >= x2 or y1 >= y2:
                            logging.warning(f"Invalid bbox {i}_{j} in {img_path}: {x1},{y1},{x2},{y2}")
                            continue

                        cropped_img = original_img[y1:y2, x1:x2]
                        if cropped_img.size == 0:
                            logging.warning(f"Empty crop in {img_path} for bbox {i}_{j}")
                            continue

                        # Создаём файл
                        output_name = f"{source_filename}_{class_name}_{i}_{j}_conf{box.conf[0]:.2f}.{IMG_FORMAT}"
                        output_path = output_dir / output_name

                        # Save the cropped image
                        cv2.imwrite(str(output_path), cropped_img,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 100] if IMG_FORMAT == 'jpg' else [])
                        logging.info(f"Saved word {i}_{j} to {output_path}")

                    except Exception as e:
                        logging.error(f'Error processing box {i}_{j} in {img_path}: {str(e)}')

            # Save annotated image
            annotated_path = output_dir / f"{source_filename}_annotated.{IMG_FORMAT}"
            cv2.imwrite(str(annotated_path), annotated_img)
            logging.info(f'Saved annotated: {annotated_path}')

def main():

    main0()
    # Конфигурация
    config = Config()

    # Пути (замените на свои)
    model_path = "models/best_model-0.6780.pt"  # Укажите путь к вашей модели
    images_folder = "data/cropped_boxes"  # Укажите путь к папке с изображениями
    output_file = "predictions.json"  # Файл для сохранения результатов

    # Инициализация предсказателя
    predictor = OcrPredictor(model_path, config)

    # Получение предсказаний
    predictions = predictor.predict_from_folder(images_folder)

    # Функция для сортировки файлов по номеру строки и слова
    def sort_key(filename):
        """Функция для сортировки файлов по номеру строки и слова"""
        try:
            # Разделяем имя файла по подчеркиваниям
            parts = filename.split('_')
            # Ищем индекс 'textline'
            if 'textline' in parts:
                textline_idx = parts.index('textline')
                # Номер строки — следующее значение после 'textline'
                line_num = int(parts[textline_idx + 1])
                # Номер слова — значение после номера строки
                word_num = int(parts[textline_idx + 2].split('.')[0].split('conf')[0])
                return (line_num, word_num)
            return (float('inf'), float('inf'))  # Некорректные файлы в конец
        except (ValueError, IndexError):
            return (float('inf'), float('inf'))  # Обработка ошибок

    # Сортируем предсказания
    sorted_predictions = sorted(
        predictions.items(),
        key=lambda x: sort_key(x[0])
    )

    # Сохранение результатов
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted_predictions), f, ensure_ascii=False, indent=2)

    # Вывод только текста в правильном порядке
    print("\nРезультаты распознавания (отсортированные по строкам и словам):")
    print("=" * 50)

    current_line = None
    for filename, text in sorted_predictions:
        try:
            parts = filename.split('_')
            if 'textline' in parts:
                textline_idx = parts.index('textline')
                line_num = int(parts[textline_idx + 1])
                if line_num != current_line:
                    if current_line is not None:
                        print()  # Новая строка для следующей строки текста
                    print(f"Строка {line_num}: ", end="")
                    current_line = line_num
                print(text, end=" ")
        except (ValueError, IndexError):
            continue  # Пропускаем файлы с некорректными именами

    print("\n" + "=" * 50)
    print(f"\nПолные результаты сохранены в файл: {output_file}")

if __name__ == '__main__':
    start_time = time.time()  # Засекаем время начала выполнения

    print("Запуск программы...")
    main()

    end_time = time.time()  # Засекаем время окончания
    elapsed_time = end_time - start_time

    # Преобразуем время в удобный формат
    elapsed_time_str = str(timedelta(seconds=elapsed_time))
    print(f"\nПрограмма завершена. Общее время выполнения: {elapsed_time_str}")