import json

from typing import Dict, Any, Tuple


def check_right_format(
        current_score: float,
        avg_previous_scores: float,
        watched_video: float,
        discussion_posts: int,
        met_deadlines: bool,
        preparation_level: str
) -> None:
    """
    Функция проверяет правильность начальных данных.

    :raises ValueError: Если входные данные некорректны
    """
    # Текущий % баллов (0 - 100)
    if not (0 <= current_score <= 100):
        raise ValueError(f"current_score должен быть в диапазоне 0-100, получено {current_score}")

    # Средний балл по предыдущим тестам (0 - 100)
    if not (0 <= avg_previous_scores <= 100):
        raise ValueError(f"avg_previous_scores должен быть в диапазоне 0-100, получено {avg_previous_scores}")

    # Просмотр видео (0 - 100)
    if not (0 <= watched_video <= 100):
        raise ValueError(f"watched_video должен быть в диапазоне 0-100, получено {watched_video}")

    # Участие в дискуссиях (Кол-во, int)
    if not isinstance(discussion_posts, int) or discussion_posts < 0:
        raise ValueError(f"discussion_posts должно быть целым числом ≥0, получено {discussion_posts}")

    # Соблюдение дедлайнов (True/False)
    if not isinstance(met_deadlines, bool):
        raise ValueError(f"met_deadlines должно быть True/False, получено {met_deadlines}")

    # Уровень подготовки (high/medium/low)
    if preparation_level not in {"high", "medium", "low"}:
        raise ValueError(f"preparation_level должен быть 'high', 'medium' или 'low', получено {preparation_level}")


def normalize(
        current_score: float,
        avg_previous_scores: float,
        watched_video: float,
        discussion_posts: int,
        met_deadlines: bool,
        preparation_level: str
) -> Dict[str, float]:
    """
    Функция нормализует входные данные, приводя их в диапазон 0-1.

    :param current_score: Текущий % баллов (0 - 100)
    :param avg_previous_scores: Средний балл по предыдущим тестам (0 - 100)
    :param watched_video: Просмотр видео (0 - 100)
    :param discussion_posts: Участие в дискуссиях (Кол-во)
    :param met_deadlines: Соблюдение дедлайнов (True/False)
    :param preparation_level: Уровень подготовки (high/medium/low)
    :return: Словарь с нормализованными весами
    """
    # Пороги
    THRESHOLD_CURRENT = 70  # Порог для текущих баллов
    THRESHOLD_AVG = 65  # Порог для среднего балла
    THRESHOLD_VIDEO = 75  # Порог для просмотра видео
    THRESHOLD_DISCUSSION = 3  # Минимальное кол-во участия в дискуссиях

    # Нормализация текущих баллов
    normalized_current = min(current_score / THRESHOLD_CURRENT, 1.0)

    # Нормализация среднего балла
    normalized_avg = min(avg_previous_scores / THRESHOLD_AVG, 1.0)

    # Бинарные признаки
    normalized_video = 1.0 if watched_video >= THRESHOLD_VIDEO else 0.0
    normalized_discussion = 1.0 if discussion_posts >= THRESHOLD_DISCUSSION else 0.0
    normalized_deadlines = 1.0 if met_deadlines else 0.0

    # Уровень подготовки (категориальный)
    if preparation_level == "high":
        normalized_level = 1.0
    elif preparation_level == "medium":
        normalized_level = 0.5
    elif preparation_level == "low":
        normalized_level = 0.0

    return {
        "current_score": normalized_current,
        "avg_previous_scores": normalized_avg,
        "watched_video": normalized_video,
        "discussion_posts": normalized_discussion,
        "met_deadlines": normalized_deadlines,
        "preparation_level": normalized_level
    }


def predict_success(
        student_data: Dict[str, Any],
        threshold: float = 0.70
) -> Tuple[str, float]:
    """
    Предсказывает успешность освоения темы на основе входных данных.

    :param student_data: Словарь с данными студента
    :param threshold: Порог для определения успешности (по умолчанию 0.70)
    :return: Кортеж (результат, вероятность)
    """
    # Проверка формата входных данных
    check_right_format(**student_data)

    # Веса для расчёта
    weights = {
        'current_score': 0.30,              # Текущий % баллов
        'avg_previous_scores': 0.20,        # Средний балл по предыдущим тестам
        'watched_video': 0.15,              # Просмотр видео
        'discussion_posts': 0.10,           # Участие в дискуссиях
        'met_deadlines': 0.15,              # Соблюдение дедлайнов
        'preparation_level': 0.10           # Уровень подготовки
    }

    # Нормализация признаков
    normalized = normalize(**student_data)

    # Расчет P(success)
    p_success = sum(normalized[feature] * weights[feature] for feature in weights)

    return "successful" if p_success >= threshold else "unsuccessful", p_success


def predict_success_from_json(json_data: str, threshold: float = 0.70) -> str:
    """
    Предсказывает успешность освоения темы на основе JSON-строки.

    :param json_data: JSON-строка с данными студента
    :param threshold: Порог для определения успешности (по умолчанию 0.70)
    :return: JSON-строка с результатом
    """
    try:
        # Парсинг JSON
        student_data = json.loads(json_data)

        # Проверка наличия всех полей
        required_fields = [
            'current_score', 'avg_previous_scores', 'watched_video',
            'discussion_posts', 'met_deadlines', 'preparation_level'
        ]
        for field in required_fields:
            if field not in student_data:
                raise ValueError(f"Отсутствует обязательное поле: {field}")

        # Предсказание
        result, p_success = predict_success(student_data, threshold)

        # Формирование результата в JSON
        response = {
            "result": result,
            "probability": round(p_success, 4),
        }
        return json.dumps(response, ensure_ascii=False, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Ошибка парсинга JSON: {str(e)}"}, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Неизвестная ошибка: {str(e)}"}, ensure_ascii=False)


def tests():
    # Успешный
    student_a = {
        "current_score": 85,
        "avg_previous_scores": 75,
        "watched_video": 90,
        "discussion_posts": 5,
        "met_deadlines": True,
        "preparation_level": "high"
    }

    # Пограничный, почти успешный
    student_b = {
        "current_score": 68,
        "avg_previous_scores": 68,
        "watched_video": 74,
        "discussion_posts": 2,
        "met_deadlines": True,
        "preparation_level": "medium"
    }

    # Неуспешный из-за лени
    student_c = {
        "current_score": 50,
        "avg_previous_scores": 40,
        "watched_video": 30,
        "discussion_posts": 0,
        "met_deadlines": False,
        "preparation_level": "low"
    }

    # Неуспешный из-за просрочек
    student_d = {
        "current_score": 80,
        "avg_previous_scores": 70,
        "watched_video": 100,
        "discussion_posts": 10,
        "met_deadlines": False,
        "preparation_level": "high"
    }

    # Успешный «середнячок»
    student_e = {
        "current_score": 72,
        "avg_previous_scores": 65,
        "watched_video": 80,
        "discussion_posts": 3,
        "met_deadlines": True,
        "preparation_level": "medium"
    }

    students = [student_a, student_b, student_c, student_d, student_e]
    for student in students:
        result, p = predict_success(student)
        print(f"Результат: {result}, Вероятность Успеха: {p:.2f}")


def main():
    json_input = '''
    {
        "current_score": 85.5,
        "avg_previous_scores": 60,
        "watched_video": 80,
        "discussion_posts": 2,
        "met_deadlines": true,
        "preparation_level": "medium"
    }
    '''

    # Вызов функции
    json_output = predict_success_from_json(json_input)
    print(json_output)


if __name__ == "__main__":
    #tests()
    main()
