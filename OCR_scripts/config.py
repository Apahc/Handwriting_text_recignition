import torch


class Config:
    def __init__(self):
        self.alphabet = " абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        self.image_width = 256
        self.image_height = 32
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
