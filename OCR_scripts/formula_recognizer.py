import cv2
import numpy as np
import logging
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import subprocess
import os
import shutil


class FormulaRecognizer:
    def __init__(self, model_name="fhswf/TrOCR_Math_handwritten"):
        self.processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def segment_lines(self, image_input, output_dir, img_format='png',
                      threshold=200, min_line_height=10, min_gap_height=5, is_bytes=False):
        """Сегментация строк из файла или байтов"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if is_bytes:
                nparr = np.frombuffer(image_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                source_filename = "image_from_bytes"
            else:
                img_path = Path(image_input)
                img = cv2.imread(str(img_path))
                source_filename = img_path.stem

            if img is None:
                raise ValueError("Не удалось загрузить изображение")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=50)
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

            projection = np.sum(binary, axis=1) / 255
            height = img.shape[0]
            line_bboxes = []
            start_y = None

            for y in range(height):
                if projection[y] > 0 and start_y is None:
                    start_y = y
                elif projection[y] == 0 and start_y is not None:
                    if y - start_y >= min_line_height:
                        line_bboxes.append((0, start_y, img.shape[1], y))
                    start_y = None

            if start_y is not None and height - start_y >= min_line_height:
                line_bboxes.append((0, start_y, img.shape[1], height))

            filtered_bboxes = []
            prev_y2 = -min_gap_height - 1
            for (x1, y1, x2, y2) in line_bboxes:
                if y1 - prev_y2 >= min_gap_height:
                    filtered_bboxes.append((x1, y1, x2, y2))
                prev_y2 = y2

            line_paths = []
            annotated_img = img.copy()
            for i, (x1, y1, x2, y2) in enumerate(filtered_bboxes):
                try:
                    cropped_img = img[y1:y2, x1:x2]
                    if cropped_img.size == 0:
                        continue

                    output_name = f"{source_filename}_line_{i}.{img_format}"
                    output_path = output_dir / output_name
                    cv2.imwrite(str(output_path), cropped_img)
                    line_paths.append(str(output_path))
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except Exception as e:
                    logging.error(f'Error processing line {i}: {str(e)}')

            annotated_path = output_dir / f"{source_filename}_annotated.{img_format}"
            cv2.imwrite(str(annotated_path), annotated_img)
            return line_paths

        except Exception as e:
            logging.error(f'Segmentation error: {str(e)}')
            raise

    def recognize_formula(self, image_path):
        """Распознавание формулы из файла"""
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        generated_ids = self.model.generate(
            pixel_values,
            max_length=100,
            num_beams=10,
            early_stopping=True
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    @staticmethod
    def create_latex_pdf(latex_code, output_file="formula.pdf"):
        """Создание PDF из LaTeX кода"""
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
        subprocess.run(["pdflatex", "temp.tex"], check=True)
        os.rename("temp.pdf", output_file)

    @staticmethod
    def cleanup(output_dir=None):
        """Очистка временных файлов"""
        for temp_file in ["temp.tex", "temp.aux", "temp.log"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
