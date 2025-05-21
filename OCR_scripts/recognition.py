import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np

import os
import json
import requests
import shutil

from typing import List, Dict, Tuple


OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'


class Tokenizer:
    def __init__(self, alphabet: str):
        self.char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
        self.char_map[CTC_BLANK] = 0
        self.char_map[OOV_TOKEN] = 1
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    def decode(self, enc_word_list: List[List[int]]) -> List[str]:
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

    def get_num_chars(self) -> int:
        return len(self.char_map)


class ImageResize:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)


class Normalize:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32) / 255


class ToTensor:
    def __call__(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr)


class MoveChannels:
    def __init__(self, to_channels_first: bool = True):
        self.to_channels_first = to_channels_first

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        return np.moveaxis(image, 0, -1)


def get_val_transforms(height: int, width: int) -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose([
        ImageResize(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])


def get_resnet34_backbone(weights: bool = True) -> nn.Sequential:
    # Получаем веса для ResNet34
    weights = torchvision.models.ResNet34_Weights.DEFAULT if weights else None

    # Загружаем модель с указанными весами
    m = torchvision.models.resnet34(weights=weights)

    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class CRNN(nn.Module):
    def __init__(self, number_class_symbols: int, time_feature_count: int = 256, lstm_hidden: int = 256,
                 lstm_len: int = 2):
        super().__init__()
        self.feature_extractor = get_resnet34_backbone(weights=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x


class OCRPredictor:
    def __init__(self, model_path: str, config):
        self.config = config
        self.tokenizer = Tokenizer(config.alphabet)
        self.device = torch.device(config.device)

        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transforms = get_val_transforms(config.image_height, config.image_width)

    def predict_batch(self, images: List[np.ndarray]) -> List[str]:
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
        return self.predict_batch([image])[0]

    def predict_from_folder(self, folder_path: str, batch_size: int = 32) -> Dict[str, str]:
        results = {}
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)

        all_predictions = []
        all_files = []

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

        if all_predictions:
            corrected_texts = [self._correct_text_with_yandex_speller(pred) for pred in all_predictions]
            for img_file, corrected_pred in zip(all_files, corrected_texts):
                results[img_file] = corrected_pred

        try:
            shutil.rmtree(folder_path)  # Удаляет папку и все ее содержимое
            print(f"Папка '{folder_path}' успешно удалена.")
        except OSError as e:
            print(f"Ошибка при удалении папки '{folder_path}': {e}")

        return results

    @staticmethod
    def _correct_text_with_yandex_speller(text: str) -> str:
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

    @staticmethod
    def sort_predictions(predictions: Dict[str, str]) -> List[Tuple[str, str]]:
        def sort_key(filename):
            try:
                parts = filename.split('_')
                if 'textline' in parts:
                    textline_idx = parts.index('textline')
                    line_num = int(parts[textline_idx + 1])
                    word_num = int(parts[textline_idx + 2].split('.')[0].split('conf')[0])
                    return line_num, word_num
                return float('inf'), float('inf')
            except (ValueError, IndexError):
                return float('inf'), float('inf')

        return sorted(predictions.items(), key=lambda x: sort_key(x[0]))

    @staticmethod
    def save_results_to_json(predictions: Dict[str, str], output_file: str) -> None:
        sorted_predictions = OCRPredictor.sort_predictions(predictions)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dict(sorted_predictions), f, ensure_ascii=False, indent=2)

    @staticmethod
    def get_and_print_results(predictions: Dict[str, str]) -> str:
        res_str = []
        curr_line = []

        sorted_predictions = OCRPredictor.sort_predictions(predictions)

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
                            print()
                            res_str.append(" ".join(curr_line))
                            curr_line.clear()
                        print(f"Строка {line_num}: ", end="")
                        current_line = line_num
                    curr_line.append(text)
                    print(text, end=" ")
            except (ValueError, IndexError):
                continue

        print("\n" + "=" * 50)
        res_str.append(" ".join(curr_line))
        return "\n".join(res_str)
