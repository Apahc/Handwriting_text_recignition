import cv2
import logging
from pathlib import Path
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO


class ImageSegmenter:
    def __init__(
            self,
            model_path: str = 'models/model.pt',
            conf_threshold: float = 0.3,
            overlap_threshold: float = 0.35,
            scale_coeff: int = 2,
            scale_bbox: float = 0.01,
            space_threshold_coeff: float = 0.0025,
            min_space_width: float = 0.02
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.overlap_threshold = overlap_threshold
        self.scale_coeff = scale_coeff
        self.scale_bbox = scale_bbox
        self.space_threshold_coeff = space_threshold_coeff
        self.min_space_width = min_space_width
        logging.info(f'Segmentation model "{model_path}" loaded')

    @staticmethod
    def preprocess_image(image, scale_coeff: int):
        """Растягиваем картинку в оттенках серого по Y"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray, None, fx=1, fy=scale_coeff, interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def calculate_intersection(box_a, box_b):
        """Вычисляет площадь пересечения двух bounding boxes"""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def filter_boxes(self, boxes):
        """Фильтрует дублирующиеся bounding boxes"""
        filtered = []
        for current_box in sorted(boxes, key=lambda x: x.conf, reverse=True):
            current_coords = current_box.xyxy[0].int().tolist()
            current_area = (current_coords[2] - current_coords[0]) * (current_coords[3] - current_coords[1])

            keep = True
            for idx, kept_box in enumerate(filtered):
                kept_coords = kept_box.xyxy[0].int().tolist()
                kept_area = (kept_coords[2] - kept_coords[0]) * (kept_coords[3] - kept_coords[1])
                inter_area = self.calculate_intersection(current_coords, kept_coords)

                if inter_area > self.overlap_threshold * current_area and inter_area > self.overlap_threshold * kept_area:
                    keep = False
                    break

            if keep:
                filtered.append(current_box)

        return filtered

    def get_word_bboxes(self, img, x1, y1, x2, y2):
        """Выделяет bounding boxes для отдельных слов в строке"""
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        height = closed.shape[0]
        center_start = height // 2 - 2
        center_end = height // 2 + 10
        center_region = closed[center_start:center_end, :]

        projection = np.sum(center_region, axis=0)
        projection = projection * (height / 12)
        space_threshold = self.space_threshold_coeff * height * 255

        space_indices = np.where(projection < space_threshold)[0]
        spaces = []
        if len(space_indices) > 0:
            start = space_indices[0]
            end = start
            for idx in space_indices[1:]:
                if idx - end == 1:
                    end = idx
                else:
                    if (end - start + 1) >= (x2 - x1) * self.min_space_width:
                        spaces.append((start, end))
                    start = end = idx
            if (end - start + 1) >= (x2 - x1) * self.min_space_width:
                spaces.append((start, end))

        word_boxes = []
        prev_end = 0
        for (s_start, s_end) in spaces:
            if s_start > prev_end:
                word_boxes.append((prev_end, 0, s_start, closed.shape[0]))
            prev_end = s_end

        if prev_end < closed.shape[1]:
            word_boxes.append((prev_end, 0, closed.shape[1], closed.shape[0]))

        width = x2 - x1
        min_word_width = width * 0.01
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

    @staticmethod
    def sort_word_bboxes(word_bboxes: List[Tuple[int, int, int, int]], line_overlap_threshold: float = 0.7
                         ) -> List[Tuple[int, int, int, int]]:
        """Сортирует bounding boxes по строкам и внутри строк"""
        bboxes_with_height = [(box, box[3] - box[1]) for box in word_bboxes]
        sorted_boxes = sorted(bboxes_with_height, key=lambda x: x[0][1])

        lines = []
        for box, h in sorted_boxes:
            y_center = (box[1] + box[3]) / 2
            matched = False

            for line in lines:
                ref_box = line[0][0]
                line_height = ref_box[3] - ref_box[1]
                threshold = line_height * line_overlap_threshold

                if ref_box[1] - threshold <= y_center <= ref_box[3] + threshold:
                    line.append((box, h))
                    matched = True
                    break

            if not matched:
                lines.append([(box, h)])

        result = []
        for line in lines:
            line_sorted = sorted(line, key=lambda x: x[0][0])
            result.extend([box for box, h in line_sorted])

        return result

    def process_image(self, img_input: Path | bytes, output_dir: Path, img_format: str = 'png',
                      is_bytes: bool = False) -> None:
        """Обрабатывает одно изображение и сохраняет результаты"""
        try:
            if is_bytes:
                # Обработка байтов
                nparr = np.frombuffer(img_input, np.uint8)
                original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                source_filename = "image_from_bytes"
            else:
                img_path = Path(img_input)
                original_img = cv2.imread(str(img_path))
                source_filename = img_path.stem

            if original_img is None:
                raise ValueError("Не удалось декодировать изображение из байтов")

            processed_img = self.preprocess_image(original_img, self.scale_coeff)

            results = self.model.predict(source=processed_img, iou=0.2, agnostic_nms=True)

            for result in results:
                annotated_img = original_img.copy()

                boxes = [box for box in result.boxes if box.conf >= self.conf_threshold]
                filtered_boxes = self.filter_boxes(boxes)

                sorted_boxes = sorted(filtered_boxes,
                                      key=lambda b: (
                                          b.xyxy[0][1].item(),
                                          b.xyxy[0][0].item()
                                      ))

                for i, box in enumerate(sorted_boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0].item())
                    class_name = self.model.names.get(class_id, 'unknown')
                    y1 //= self.scale_coeff
                    y2 //= self.scale_coeff

                    word_bboxes = self.get_word_bboxes(annotated_img, x1, y1, x2, y2)
                    word_bboxes = self.sort_word_bboxes(word_bboxes)

                    for j, (x1, y1, x2, y2) in enumerate(word_bboxes):
                        try:
                            h, w = original_img.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)

                            if x1 >= x2 or y1 >= y2:
                                logging.warning(f"Invalid bbox {i}_{j} in {img_input}: {x1},{y1},{x2},{y2}")
                                continue

                            cropped_img = original_img[y1:y2, x1:x2]
                            if cropped_img.size == 0:
                                logging.warning(f"Empty crop in {img_input} for bbox {i}_{j}")
                                continue

                            output_name = f"{source_filename}_{class_name}_{i}_{j}_conf{box.conf[0]:.2f}.{img_format}"
                            output_path = output_dir / output_name

                            cv2.imwrite(str(output_path), cropped_img,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 100] if img_format == 'jpg' else [])
                            logging.info(f"Saved word {i}_{j} to {output_path}")

                        except Exception as e:
                            logging.error(f'Error processing box {i}_{j} in {img_input}: {str(e)}')

                annotated_path = f"./data/image_bboxes/{source_filename}_annotated.{img_format}"
                cv2.imwrite(str(annotated_path), annotated_img)
                logging.info(f'Saved annotated: {annotated_path}')
        except Exception as e:
            if is_bytes:
                logging.error(f'Error processing image bytes: {str(e)}')
            else:
                logging.error(f'Error processing image path: {str(e)}')
            raise
