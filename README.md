# Handwriting Text Recognition 🌸

Добро пожаловать в проект **Handwriting Text Recognition**! 🧚‍♀️ Этот проект предназначен для распознавания рукописного текста на русском и английском языках, а также математических выражений на изображениях с последующим переводом распознанного текста на другие языки. Кроме того, проект включает модули для оценки успешности студентов и проверки текстовых ответов на соответствие эталону. 💖

Проект размещен на GitHub: [Apahc/Handwriting_text_recognition](https://github.com/Apahc/Handwriting_text_recignition). 🌸

## Описание проекта 💖

Проект объединяет несколько модулей для обработки рукописного текста и анализа учебных данных:

1. **Распознавание русского рукописного текста** (`text_recognition.py`): Использует модель CRNN с ResNet34 для распознавания кириллического текста.
2. **Распознавание математических выражений** (`formula_recognizer.py`): Преобразует формулы в LaTeX и создает PDF.
3. **Распознавание английского рукописного текста** (`trocr_script.py`): Использует предобученную модель TrOCR для обработки англоязычного текста.
4. **Оценка успешности студентов** (`student_success_predictor.py`): Прогнозирует успех освоения темы на основе академических и поведенческих характеристик.
5. **Проверка текстовых ответов** (`AnswerGrader`): Сравнивает ответы студентов с эталоном с помощью fuzzy matching и семантического анализа (SBERT).

## Оглавление 🌸

- [Установка](#установка) 🧚‍♀️
- [Структура проекта](#структура-проекта) 💖
- [Использование](#использование) 🌸
- [Описание модулей](#описание-модулей) 🧚‍♀️
  - [Распознавание русского текста](#распознавание-русского-текста) 💖
  - [Распознавание математических выражений](#распознавание-математических-выражений) 🌸
  - [Распознавание английского текста](#распознавание-английского-текста) 🧚‍♀️
  - [Оценка успешности студентов](#оценка-успешности-студентов) 💖
  - [Проверка текстовых ответов](#проверка-текстовых-ответов) 🌸

## Установка 🧚‍♀️

Для работы проекта установите зависимости и подготовьте окружение:

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/Apahc/Handwriting_text_recignition.git
   cd Handwriting_text_recignition
   ```

2. **Установите Python 3.8+** и создайте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```

   Содержимое `requirements.txt`:
   ```
   torch>=2.0.0
   torchvision>=0.15.0
   opencv-python>=4.5.0
   numpy>=1.21.0
   requests>=2.28.0
   ultralytics>=8.0.0
   fuzzywuzzy>=0.18.0
   sentence-transformers>=2.2.0
   razdel>=0.5.0
   transformers>=4.30.0
   pillow>=9.0.0
   ```

4. **Скачайте модели**:
   - Для русского текста: `models/model.pt` (YOLOv8) и `models/best_model-0.6780.pt` (CRNN).
   - Для английского текста: `models/trocr-base-handwritten` (TrOCR).
   - Для проверки ответов: `paraphrase-multilingual-MiniLM-L12-v2` (автоматически загружается SentenceTransformer).

5. **Подготовьте данные**:
   - Создайте папки `data/my_test`, `data/math_test`, `data/english_test` для входных изображений.
   - Создайте `data/cyrillic_handwriting` с TSV-файлами (`train.tsv`, `test.tsv`) для русского текста.

## Структура проекта 💖

```plaintext
Handwriting_text_recignition/
├── data/
│   ├── my_test/                # Изображения для русского текста
│   ├── math_test/              # Изображения для формул
│   ├── english_test/           # Изображения для английского текста
│   ├── cropped_boxes/          # Вырезанные слова (русский текст)
│   ├── cropped_lines/          # Вырезанные строки (формулы, английский текст)
│   ├── image_bboxes/           # Аннотированные изображения
│   └── cyrillic_handwriting/   # Датасет для русского текста
├── models/                     # Модели (YOLOv8, CRNN, TrOCR)
├── text_recognition.py         # Распознавание русского текста
├── formula_recognizer.py       # Распознавание формул
├── trocr_script.py             # Распознавание английского текста
├── student_success_predictor.py # Оценка успешности студентов
├── AnswerGrader.py             # Проверка текстовых ответов
├── requirements.txt            # Зависимости
└── README.md                   # Документация
```

## Использование 🌸

1. **Русский текст**:
   ```bash
   python text_recognition.py
   ```
   Укажите `SOURCE_PATH` (например, `data/my_test/img20.jpg`) в коде. Результаты сохраняются в `predictions.json` и выводятся в консоль.

2. **Математические выражения**:
   ```bash
   python formula_recognizer.py
   ```
   Укажите `SOURCE_PATH` (например, `data/math_test/img6.jpg`). Вывод: LaTeX-код в консоли и PDF.

3. **Английский текст**:
   ```bash
   python trocr_script.py
   ```
   Укажите `SOURCE_PATH` (например, `data/english_test/img1.jpg`). Вывод: текст в консоли.

4. **Оценка успешности студентов**:
   ```bash
   python student_success_predictor.py
   ```
   Передайте JSON с данными студента (например, `current_score`, `avg_previous_scores`). Вывод: JSON с результатом (`successful`/`unsuccessful`) и вероятностью.

5. **Проверка текстовых ответов**:
   ```python
   from AnswerGrader import AnswerGrader
   grader = AnswerGrader()
   student_answer = "Ваши ответы здесь"
   reference = grader.get_reference_answer()
   is_correct, method = grader.grade_answer(student_answer, reference)
   print(f"Правильный: {is_correct}, Метод: {method}")
   ```

## Описание модулей 🧚‍♀️

### Распознавание русского текста 💖

- **Загрузка**: Изображение из `SOURCE_PATH` или байтовый массив. Проверяется корректность файла.
- **Предобработка**:
  - Gaussian Blur (ядро 5x5) для снижения шума.
  - Адаптивная бинаризация (Otsu).
  - Вертикальное масштабирование (`SCALE_COEFF=2`, `INTER_LINEAR`).
- **Сегментация**:
  - YOLOv8 (`models/model.pt`) для bounding boxes (`CONF_THRESHOLD=0.3`).
  - Фильтрация пересечений (`OVERLAP_THRESHOLD=0.35`).
  - Разбиение на слова: морфологическая операция (ядро 5x3), проекционный профиль (`SPACE_THRESHOLD_COEFF=0.0025`, `MIN_SPACE_WIDTH=0.02`).
  - Сортировка: группировка по строкам (`line_overlap_threshold=0.7`), сортировка по x-координате.
  - Сохранение: слова в `data/cropped_boxes`, аннотации в `data/image_bboxes`.
- **Нормализация**: Ресайзинг (`Config.image_height`, `Config.image_width`), нормализация [0, 1], тензоры PyTorch (HWC→CHW).
- **Модель**:
  - CRNN: ResNet34, AdaptiveAvgPool2d (`time_feature_count=256`), BiLSTM (`hidden_size=256`, 2 слоя, dropout 0.1), классификатор (GELU, dropout 0.1).
  - CTC Loss, точность 0.6780.
  - Датасет: `data/cyrillic_handwriting` (TSV-файлы).
- **Распознавание**:
  - Пакетная обработка (`batch_size=32`).
  - CTC-декодер (`Tokenizer`) удаляет `<BLANK>` и дубликаты.
- **Постобработка**:
  - Yandex Speller API (`lang=ru`) для коррекции.
  - Фильтрация слов (`is_known_word`): исключаются смешанные кириллица/латиница, слова с цифрами и кириллицей, повторяемость <50%.
  - Сохранение: JSON (`predictions.json`), консольный вывод.

### Распознавание математических выражений 🌸

- **Сегментация**: YOLOv8 выделяет строки формул (`data/math_test/img6.jpg`), сохраняет в `data/cropped_lines`, сортирует по y-координате.
- **Нормализация**: Ресайзинг, нормализация [0, 1].
- **Модель**: `FormulaRecognizer` (предположительно CRNN/Transformer), обучена на датасете (например, CROHME), возвращает LaTeX.
- **Распознавание**: CTC-декодер или LaTeX-декодер. Вывод: список LaTeX-выражений.
- **Создание PDF**: Объединение выражений (`\\`), генерация PDF (`create_latex_pdf`).
- **Сохранение**: LaTeX в консоли, PDF, аннотации в `data/image_bboxes`.

### Распознавание английского текста 🧚‍♀️

- **Загрузка**: `data/english_test/img1.jpg`.
- **Предобработка**: Оттенки серого, контрастность (`alpha=2.0`, `beta=50`), бинаризация (`THRESHOLD=200`).
- **Сегментация**: Горизонтальная проекция, строки высотой ≥ `MIN_LINE_HEIGHT=30`, зазоры ≥ `MIN_GAP_HEIGHT=5`. Сохранение в `data/cropped_lines`.
- **Нормализация**: RGB через PIL, тензоры через `TrOCRProcessor`.
- **Модель**: `VisionEncoderDecoderModel` (`trocr-base-handwritten`), предобучена на IAM.
- **Распознавание**: Генерация текста, декодирование через `TrOCRProcessor`. Вывод: консоль (строки разделены переносами).

### Оценка успешности студентов 💖

- **Проверка данных**: `current_score`, `avg_previous_scores`, `watched_video` (0–100), `discussion_posts` (целое ≥0), `met_deadlines` (bool), `preparation_level` (`high`, `medium`, `low`).
- **Нормализация**:
  - `current_score`: делится на 70, ограничено 1.0.
  - `avg_previous_scores`: делится на 65, ограничено 1.0.
  - `watched_video`: 1.0 (≥75), иначе 0.0.
  - `discussion_posts`: 1.0 (≥3), иначе 0.0.
  - `met_deadlines`: 1.0 (`True`), иначе 0.0.
  - `preparation_level`: `high` (1.0), `medium` (0.5), `low` (0.0).
- **Предсказание**: Взвешенная сумма (веса: 0.3, 0.2, 0.15, 0.1, 0.15, 0.1). Результат: `successful` (≥0.7), иначе `unsuccessful`.
- **Вывод**: JSON с результатом и вероятностью.

### Проверка текстовых ответов 🌸

- **Инициализация**: `SentenceTransformer` (`paraphrase-multilingual-MiniLM-L12-v2`).
- **Нормализация**: Нижний регистр, `ё`→`е`, замена символов (`²`→`2`, `×`→`x`), удаление мусора или лишних символов.
- **Разбиение**: `razdel.sentenize` или регулярное выражение (`[.!?…]\s+`), фильтрация предложений (<5 символов).
- **Сравнение**:
  - Точное совпадение нормализованных текстов.
  - Fuzzy matching (`fuzz.ratio`, порог 80): полный текст, затем по предложениям.
  - SBERT: косинусная близость (порог 0.8) для полного текста или предложений (`all`/`any`).
- **Оценка**: Последовательная проверка (точное → fuzzy → SBERT). Возвращает `(is_correct, method)`.
- **Эталон**: Текст о последствиях глобального потепления.


© 2025 Apahc, fantas-coder. Все права защищены. 🧚‍♀️
