from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import subprocess
import os

# Загрузка процессора и модели
try:
    processor = TrOCRProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten", use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit()

# Установка устройства (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Функция для обработки изображения
def recognize_formula(image_path):
    # Загрузка изображения
    image = Image.open(image_path).convert("RGB")

    # Подготовка изображения для модели
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Генерация LaTeX с улучшенными параметрами
    generated_ids = model.generate(
        pixel_values,
        max_length=100,
        num_beams=10,
        early_stopping=True,
        temperature=0.7
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

# Функция для создания PDF
def create_latex_pdf(latex_code, output_file="formula.pdf"):
    latex_template = r"""
    \documentclass{article}
    \usepackage{amsmath}
    \begin{document}
    \begin{equation}
    %s
    \end{equation}
    \end{document}
    """
    with open("temp.tex", "w", encoding="utf-8") as f:
        f.write(latex_template % latex_code)
    subprocess.run(["pdflatex", "temp.tex"])
    os.rename("temp.pdf", output_file)

# Пример использования
image_path = "data/math_test/img5.jpg"  # Укажите путь к улучшенному изображению
try:
    latex_output = recognize_formula(image_path)
except Exception as e:
    print(f"Ошибка обработки изображения: {e}")
    exit()

# Вывод в консоль
print("LaTeX output:", latex_output)

# Сохранение в файл
with open("output.tex", "w", encoding="utf-8") as f:
    f.write(latex_output)

# Создание PDF (опционально)
try:
    create_latex_pdf(latex_output, "formula.pdf")
except FileNotFoundError:
    print("pdflatex не найден. Установите TeXLive или MiKTeX для создания PDF.")