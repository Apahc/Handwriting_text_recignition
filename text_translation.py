from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from langdetect import detect, LangDetectException
import torch


def add_punctuation(text):
    """Добавляет точку в конец текста, если он не заканчивается знаком препинания."""
    punctuation_marks = ".!?:;"
    if not text or text[-1] not in punctuation_marks:
        return text + "."
    return text


def load_model():
    """Функция для загрузки модели и токенайзера"""
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model.eval()
    return model, tokenizer


def translate_text(text, model, tokenizer):
    """Функция для распознавания языка и перевода текста на русский."""
    # Добавляем точку в конец для лучшего распознавания
    text = add_punctuation(text)

    # Распознавание языка
    try:
        detected_lang = detect(text)
        # print(f"Обнаруженный язык: {detected_lang}")
    except LangDetectException:
        detected_lang = "en"
        # print("Не удалось определить язык, используется английский по умолчанию")

    # Установка исходного языка
    tokenizer.src_lang = lang_map.get(detected_lang, "en_XX")
    # print(f"Установлен исходный язык: {tokenizer.src_lang}")

    # Кодирование текста
    encoded = tokenizer(text, return_tensors="pt")

    # Генерация перевода на русский
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id["ru_RU"]
        )

    # Декодирование и возврат перевода
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    if detect(translation) != "ru":
        translation = translate_text(translation, model, tokenizer)

    return translation


def main():
    # Загрузка модели и токенизатора
    model, tokenizer = load_model()

    # Текст для перевода
    article_language = "La educación es la clave para un futuro mejor"

    # Вывод
    translation = translate_text(article_language, model, tokenizer)
    print(f"Исходный текст: {article_language}")
    print(f"Перевод на русский: {translation}")


lang_map = {
    "ar": "ar_AR",  # Arabic
    "cs": "cs_CZ",  # Czech
    "de": "de_DE",  # German
    "en": "en_XX",  # English
    "es": "es_XX",  # Spanish
    "et": "et_EE",  # Estonian
    "fi": "fi_FI",  # Finnish
    "fr": "fr_XX",  # French
    "gu": "gu_IN",  # Gujarati
    "hi": "hi_IN",  # Hindi
    "it": "it_IT",  # Italian
    "ja": "ja_XX",  # Japanese
    "kk": "kk_KZ",  # Kazakh
    "ko": "ko_KR",  # Korean
    "lt": "lt_LT",  # Lithuanian
    "lv": "lv_LV",  # Latvian
    "my": "my_MM",  # Burmese
    "ne": "ne_NP",  # Nepali
    "nl": "nl_XX",  # Dutch
    "ro": "ro_RO",  # Romanian
    "ru": "ru_RU",  # Russian
    "si": "si_LK",  # Sinhala
    "tr": "tr_TR",  # Turkish
    "vi": "vi_VN",  # Vietnamese
    "zh": "zh_CN",  # Chinese
    "zh-cn": "zh_CN",  # Chinese
    "af": "af_ZA",  # Afrikaans
    "az": "az_AZ",  # Azerbaijani
    "bn": "bn_IN",  # Bengali
    "fa": "fa_IR",  # Persian
    "he": "he_IL",  # Hebrew
    "hr": "hr_HR",  # Croatian
    "id": "id_ID",  # Indonesian
    "ka": "ka_GE",  # Georgian
    "km": "km_KH",  # Khmer
    "mk": "mk_MK",  # Macedonian
    "ml": "ml_IN",  # Malayalam
    "mn": "mn_MN",  # Mongolian
    "mr": "mr_IN",  # Marathi
    "pl": "pl_PL",  # Polish
    "ps": "ps_AF",  # Pashto
    "pt": "pt_XX",  # Portuguese
    "sv": "sv_SE",  # Swedish
    "sw": "sw_KE",  # Swahili
    "ta": "ta_IN",  # Tamil
    "te": "te_IN",  # Telugu
    "th": "th_TH",  # Thai
    "tl": "tl_XX",  # Tagalog
    "uk": "uk_UA",  # Ukrainian
    "ur": "ur_PK",  # Urdu
    "xh": "xh_ZA",  # Xhosa
    "gl": "gl_ES",  # Galician
    "sl": "sl_SI",  # Slovene
}


if __name__ == "__main__":
    main()
