import re
import torch
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from razdel import sentenize
from typing import Tuple


class AnswerGrader:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)

    def normalize_text(
            self,
            text: str,
            keep_custom: str = '+-x/=',
            soft_normalize: bool = False
    ) -> str:
        """
        Продвинутая нормализация с детальным контролем символов

        :param text: Исходный текст
        :param keep_custom: Символы для сохранения
        :param soft_normalize: Мягкая нормировка для SBERT
        :return: Нормализованный текст
        """
        text = text.lower().replace('ё', 'е')       # Приводим к нижнему регистру, заменяем ё на е
        replacements = {                                        # Нормализация специальных символов
            '²': '2',
            '³': '3',
            '°': ' градусов ',
            '×': 'x',
            '÷': '/',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        if soft_normalize:
            text = re.sub(r'^\s*Вариант \d+:\s*', '', text)                 # Удаляем технический мусор
        else:
            text = re.sub(fr"[^\w\s{re.escape(keep_custom)}]", '', text)    # Удаляем только специальные символы

        return re.sub(r'\s+', ' ', text).strip()                            # Удаляем повторяющиеся пробелы

    def split_text(
            self,
            text: str,
            min_sentence_length: int = 5
    ) -> list[str]:
        """
        Разбиение текста на предложения

        :param text: Исходный текст
        :param min_sentence_length: Минимальная длинна предложения
        :return: Список из разделённых частей
        """
        try:
            sentences = [s.text for s in sentenize(text)]
            if len(sentences) <= 1:
                sentences = re.split(r'(?<=[.!?…])\s+', text)
            return [s for s in sentences if len(s.strip()) >= min_sentence_length]
        except Exception as e:
            print(f"Ошибка разбиения текста: {e}")
            return [text]

    def fuzzy_match_sentences(
            self,
            student_sentences: list[str],
            reference_sentences: list[str],
            threshold: int = 80
    ) -> bool:
        """
        Сравнение предложений попарно, требуя соответствия для ВСЕХ предложений студента

        :param student_sentences: Предложения студента
        :param reference_sentences: Эталонные предложения
        :param threshold: Порог для fuzzy matching
        :return: True если для каждого предложения студента есть совпадение в эталоне
        """
        matched_reference_indices = set()

        for student_sent in student_sentences:
            found_match = False
            norm_student = self.normalize_text(student_sent)

            for i, ref_sent in enumerate(reference_sentences):
                if i in matched_reference_indices:
                    continue

                if fuzz.ratio(norm_student, self.normalize_text(ref_sent)) > threshold:
                    matched_reference_indices.add(i)
                    found_match = True
                    break

            if not found_match:
                return False

        return True

    def check_semantic_similarity(
            self,
            answer: str,
            reference: str
    ) -> float:
        """
        Сравнения текстов по семантической близости (SBERT)

        :param answer: Текст проверяемого ответа
        :param reference: Эталонный текст
        :return: Возвращаем долю близости текстов
        """
        try:
            # Кодируем тексты в векторы
            emb_answer = self.model.encode(answer, convert_to_tensor=True)
            emb_ref = self.model.encode(reference, convert_to_tensor=True)
            # Вычисляем косинусную близость
            return util.cos_sim(emb_answer, emb_ref).item()
        except Exception as e:
            print(f"Ошибка при сравнении: {e}")
            return 0.0

    def check_semantic_similarity_sentence(
            self,
            answer: str,
            reference: str,
            threshold: float = 0.8,
            strategy: str = "all"  # "any" или "all"
    ) -> bool:
        """
        Сравнивает тексты на уровне предложений

        :param answer: Предложения проверяемого ответа
        :param reference: Эталонные предложения
        :param threshold: Доля близости косинусного расстояния
        :param strategy: "any" - хотя бы одна пара предложений превышает порог
                         "all" - каждое предложение text1 имеет пару в text2
        :return: Булевое значение, отвечающее на вопрос близкие ли предложения или нет
        """
        # Получаем эмбеддинги
        emb1 = self.model.encode(answer, convert_to_tensor=True)
        emb2 = self.model.encode(reference, convert_to_tensor=True)

        # Матрица попарных схожестей [len(sentences1) x len(sentences2)]
        similarity_matrix = util.cos_sim(emb1, emb2)

        if strategy == "any":
            return torch.max(similarity_matrix) > threshold
        elif strategy == "all":
            max_similarities, _ = torch.max(similarity_matrix, dim=1)
            return torch.all(max_similarities > threshold).item()
        else:
            raise ValueError("Неизвестная стратегия: используйте 'any' или 'all'")

    def grade_answer(
            self,
            student: str,
            reference: str,
            fuzzy_threshold: float = 80,
            sbert_threshold: float = 0.8
    ) -> Tuple[bool, str]:
        """
        Функция оценивает ответ с поэтапной проверкой
        1. Точное сравнение
        2. Fuzzy Matching
        3. SBERT

        :param student: Текст проверяемого ответа
        :param reference: Эталонный текст
        :param fuzzy_threshold: Порог Fuzzy сравнения
        :param sbert_threshold: Доля SBERT сравнения
        :return: Картеж с двумя элементами: Ответ, похожие ли тексты и информация по какому критерию произошло сравнение
        """
        norm_student = self.normalize_text(student)
        norm_reference = self.normalize_text(reference)

        # 1. Точное сравнение полного текста
        if norm_student == norm_reference:
            return True, "Exact match"

        # 2. Fuzzy Matching сравнение всего текста
        fuzz_score = fuzz.ratio(norm_student, norm_reference)
        if fuzz_score > fuzzy_threshold:
            return True, f"Fuzzy ({fuzz_score}%)"

        # 3. Разбиваем на предложения
        student_sents = self.split_text(student)
        reference_sents = self.split_text(reference)

        if len(student_sents) > 1 or len(reference_sents) > 1:
            # 3.1. Fuzzy Matching с полным соответствием для предложений
            if self.fuzzy_match_sentences(student_sents, reference_sents, threshold=fuzzy_threshold):
                return True, "Fuzzy (sentences)"

            # 3.2. SBERT по предложениям
            # Жесткая проверка (все предложения)
            if self.check_semantic_similarity_sentence(student, reference, threshold=sbert_threshold, strategy="all"):
                return True, "SBERT (all)"

            # Мягкая проверка (хотя бы одно)
            if self.check_semantic_similarity_sentence(student, reference, threshold=sbert_threshold, strategy="any"):
                return True, "SBERT (any)"

        # 4. Семантическое сравнение (SBERT)
        similarity = self.check_semantic_similarity(
            self.normalize_text(student, soft_normalize=True),
            self.normalize_text(reference, soft_normalize=True)
        )
        if similarity > sbert_threshold:
            return True, f"SBERT ({similarity:.2f})"

        return False, "No match"

    def get_reference_answer(self):
        return "Конспект\nПомимо повышения уровня Мирового океана, повышение глобальной температуры также приведёт к изменениям в количестве и распределении атмосферных осадков. В результате могут участиться природные катаклизмы: наводнения, засухи, ураганы и другие. Потепление должно, по всей вероятности, увеличивать частоту и масштаб таких явлений.\nДругим возможным последствием повышения глобальных температур является снижение урожаев сельскохозяйственных культур в слаборазвитых странах Африки, Азии и Латинской Америки и повышения урожаев в развитых странах (за счёт увеличения концентрации углекислого газа и удлинения вегетативных периодов)."
