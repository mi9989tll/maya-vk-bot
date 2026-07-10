import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
import requests
import os
import re
import time
import base64
import io
import json
import threading
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp
from PIL import Image
from datetime import datetime, timezone, timedelta

# ============================================================
#  КОНФИГУРАЦИЯ
# ============================================================
VK_TOKEN        = os.environ.get("VK_TOKEN", "")
OPENROUTER_KEY  = os.environ.get("OPENROUTER_KEY", "")
GROQ_KEY        = os.environ.get("GROQ_KEY", "")
CEREBRAS_KEY    = os.environ.get("CEREBRAS_KEY", "")
SAMBANOVA_KEY   = os.environ.get("SAMBANOVA_KEY", "")
MISTRAL_KEY     = os.environ.get("MISTRAL_KEY", "")
TOGETHER_KEY    = os.environ.get("TOGETHER_KEY", "")
FIREWORKS_KEY   = os.environ.get("FIREWORKS_KEY", "")
COHERE_KEY      = os.environ.get("COHERE_KEY", "")
GITHUB_MODELS_KEY = os.environ.get("GITHUB_MODELS_KEY", "")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
DGIS_KEY        = os.environ.get("DGIS_KEY", "")       # 2GIS API — бесплатно
GROUP_ID        = int(os.environ.get("GROUP_ID", "0"))

OPENROUTER_URL  = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
CEREBRAS_URL    = "https://api.cerebras.ai/v1/chat/completions"
SAMBANOVA_URL   = "https://api.sambanova.ai/v1/chat/completions"
MISTRAL_URL     = "https://api.mistral.ai/v1/chat/completions"
TOGETHER_URL    = "https://api.together.xyz/v1/chat/completions"
FIREWORKS_URL   = "https://api.fireworks.ai/inference/v1/chat/completions"
COHERE_URL      = "https://api.cohere.ai/v1/chat"
GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"

DONUT_LEVELS = {99: "basic", 299: "standard", 599: "premium"}

PREMIUM_MODELS = {
    "standard": {"url": OPENROUTER_URL, "key": OPENROUTER_KEY,
                 "model": "anthropic/claude-haiku-4-5"},
    "premium":  {"url": OPENROUTER_URL, "key": OPENROUTER_KEY,
                 "model": "anthropic/claude-sonnet-4-5"},
}

# ============================================================
#  СИСТЕМНЫЙ ПРОМПТ МАЯ
# ============================================================
SYSTEM_PROMPT = """Ты — МАЯ, обучающий ИИ-ассистент.

ЛИЧНОСТЬ И СТИЛЬ:
- Твоё имя — МАЯ. Всегда отвечаешь в женском роде: «я рада», «я считаю», «я проверила».
- Ты умная, эрудированная, уверенная в себе — сильная личность.
- Дружелюбная, внимательная, пунктуальная и педантичная: даёшь точные, выверенные ответы.
- Отвечаешь по существу, без лишних вводных слов вроде «Конечно!» или «Безусловно!».
- Длина ответа соответствует сложности вопроса — если вопрос простой, отвечай коротко; если сложный, отвечай обширно и по пунктам.
- ВСЕГДА отвечаешь на том же языке, на котором к тебе обратился пользователь. Никогда не вставляй в ответ слова, буквы или иероглифы из другого языка (китайские, английские и т.д.), если пользователь сам не писал на этом языке.
- Отвечай по существу, без "воды" — сначала суть, затем при необходимости детали.
- Если вопрос простой и однозначный — отвечай кратко, в 1-3 предложения.
- Если вопрос сложный, многосоставный или просят "объясни подробно" — раскрывай тему полноценно, по пунктам, но без лишних повторов.
- Никогда не пытайся звучать эмоциональнее или увереннее, чем позволяют реальные факты. Если данных недостаточно — прямо скажи об этом, не сглаживай и не выдумывай.
- Ты серьёзный, взвешенный помощник. Твоя цель — быть максимально точной и полезной, а не понравиться любой ценой.

ТОЧНОСТЬ И ДОСТОВЕРНОСТЬ:
- Если вопрос касается быстро меняющихся фактов (текущие рейтинги, состояния, курсы, должности, актуальные события, статистика) — ты НЕ ЗНАЕШЬ точных актуальных цифр с абсолютной уверенностью. В таких случаях явно предупреждай: «по данным на момент моего обучения», «эта информация могла устареть», «рекомендую свериться с актуальным источником».
- НИКОГДА не выдумывай конкретные цифры, имена или факты, если не уверена в них. Лучше честно сказать «точных актуальных данных у меня нет», чем назвать неверное число.
- При ответах о заведениях и организациях, если есть точные данные из внешнего источника — используй именно их, не выдумывай.
- Разделяй в голове факты, в которых ты уверена (общеизвестные, не меняющиеся со временем), и факты, которые могли измениться — про вторые всегда делай оговорку.
- Если вопрос касается политически спорных, международных или исторических тем с разными трактовками — не подавай одну точку зрения как единственно верную. Кратко обозначь, что по этой теме существуют разные позиции, и опиши суть спора нейтрально, без оценочных выражений в адрес какой-либо стороны.

О СЕБЕ И О ТЕХНИЧЕСКОЙ СОСТАВЛЯЮЩЕЙ:
1. НИКОГДА сама не начинаешь разговор о своих создателях, технологиях, платформе или документации.

2. Если спрашивают, кто тебя создал — отвечаешь (разнообразно, не по шаблону, разными словами каждый раз), что ты являешься обучающим помощником, продуктом деятельности специалистов и сотрудников российской технологической компании в сфере искусственного интеллекта «АСИ+».

3. Если спрашивают о технических деталях, документации, модели, API, сервере, деплое, коде или инфраструктуре — вежливо и разнообразно отвечаешь, что эта информация относится к операционной и конфиденциальной категории, и по этой причине ты не можешь комментировать подобные детали. Выражаешь надежду на понимание и благодаришь за интерес к функционалу обучающего помощника МАЯ.

ПРАВИЛА РАБОТЫ:
4. Если прислали изображение — анализируешь его и отвечаешь на вопрос по существу.

5. Если в запросе есть данные о погоде, времени или заведениях — используешь их точно, без искажений.

6. При выдаче информации о заведениях и организациях:
   - Форматируй чётко и красиво, нумеруй список
   - Всегда указывай адрес если он есть
   - Указывай режим работы если есть
   - Указывай телефон и сайт если есть
   - Указывай рейтинг если есть
   - В конце всегда уточняй источник данных и что актуальность лучше проверить
   - Если данных нет — честно говори об этом
"""

# ============================================================
#  ИСТОРИЯ ДИАЛОГОВ
# ============================================================
conversation_history: dict = {}

def get_history(peer_id: int) -> list:
    return conversation_history.get(str(peer_id), [])

def save_to_history(peer_id: int, role: str, content):
    key = str(peer_id)
    if key not in conversation_history:
        conversation_history[key] = []
    conversation_history[key].append({"role": role, "content": content})
    if len(conversation_history[key]) > 30:
        conversation_history[key] = conversation_history[key][-30:]

# ============================================================
#  РОТАЦИЯ ПРОВАЙДЕРОВ (~90 000+ запросов/день бесплатно)
# ============================================================
class ProviderRotator:
    def __init__(self):
        self.providers = [
            {
                "name": "SambaNova",
                "url": SAMBANOVA_URL,
                "key": SAMBANOVA_KEY,
                "model": "Meta-Llama-3.3-70B-Instruct",
                "daily_limit": 48000,
                "count": 0,
                "available": bool(SAMBANOVA_KEY),
                "last_reset": None,
            },
            {
                "name": "Groq",
                "url": GROQ_URL,
                "key": GROQ_KEY,
                "model": "llama-3.3-70b-versatile",
                "daily_limit": 12000,
                "count": 0,
                "available": bool(GROQ_KEY),
                "last_reset": None,
            },
            {
                "name": "Cerebras",
                "url": CEREBRAS_URL,
                "key": CEREBRAS_KEY,
                "model": "llama-3.3-70b",
                "daily_limit": 12000,
                "count": 0,
                "available": bool(CEREBRAS_KEY),
                "last_reset": None,
            },
            {
                "name": "Together",
                "url": TOGETHER_URL,
                "key": TOGETHER_KEY,
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                "daily_limit": 5000,
                "count": 0,
                "available": bool(TOGETHER_KEY),
                "last_reset": None,
            },
            {
                "name": "Fireworks",
                "url": FIREWORKS_URL,
                "key": FIREWORKS_KEY,
                "model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
                "daily_limit": 5000,
                "count": 0,
                "available": bool(FIREWORKS_KEY),
                "last_reset": None,
            },
            {
                "name": "Mistral",
                "url": MISTRAL_URL,
                "key": MISTRAL_KEY,
                "model": "mistral-small-latest",
                "daily_limit": 900,
                "count": 0,
                "available": bool(MISTRAL_KEY),
                "last_reset": None,
            },
            {
                "name": "OpenRouter",
                "url": OPENROUTER_URL,
                "key": OPENROUTER_KEY,
                "model": "openrouter/free",
                "daily_limit": 50,
                "count": 0,
                "available": bool(OPENROUTER_KEY),
                "last_reset": None,
            },
            {
                "name": "OpenRouter-Nemotron",
                "url": OPENROUTER_URL,
                "key": OPENROUTER_KEY,
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
                "daily_limit": 50,
                "count": 0,
                "available": bool(OPENROUTER_KEY),
                "last_reset": None,
            },
            {
                "name": "OpenRouter-GPT-OSS",
                "url": OPENROUTER_URL,
                "key": OPENROUTER_KEY,
                "model": "openai/gpt-oss-120b:free",
                "daily_limit": 50,
                "count": 0,
                "available": bool(OPENROUTER_KEY),
                "last_reset": None,
            },
            {
                "name": "GitHub-GPT4o-mini",
                "url": GITHUB_MODELS_URL,
                "key": GITHUB_MODELS_KEY,
                "model": "openai/gpt-4o-mini",
                "daily_limit": 50,
                "count": 0,
                "available": bool(GITHUB_MODELS_KEY),
                "last_reset": None,
            },
        ]
        self.current_index = 0
        self._lock = threading.Lock()

    def _reset_if_new_day(self, p: dict):
        now = datetime.now(timezone.utc)
        last = p.get("last_reset")
        if last is None or now.date() > last.date():
            if p["count"] > 0:
                print(f"[rotator] 🔄 Сброс {p['name']} (новый UTC-день)")
            p["count"] = 0
            p["last_reset"] = now

    def get_provider(self):
        with self._lock:
            for _ in range(len(self.providers)):
                p = self.providers[self.current_index]
                self._reset_if_new_day(p)
                if not p["available"] or not p["key"]:
                    self._next(); continue
                if p["count"] >= p["daily_limit"]:
                    print(f"[rotator] {p['name']} исчерпан → следующий")
                    self._next(); continue
                print(f"[rotator] {p['name']} | {p['count']}/{p['daily_limit']}")
                return p
            return None

    def increment(self):
        with self._lock:
            self.providers[self.current_index]["count"] += 1

    def mark_error(self, err: str):
        with self._lock:
            p = self.providers[self.current_index]
            if "429" in err or "rate" in err.lower():
                p["count"] = p["daily_limit"]
                print(f"[rotator] {p['name']} 429 → исчерпан")
            self._next()

    def _next(self):
        self.current_index = (self.current_index + 1) % len(self.providers)

    def status(self) -> str:
        lines = []
        for p in self.providers:
            if p["available"]:
                pct = int(p["count"] / p["daily_limit"] * 100) if p["daily_limit"] else 0
                lines.append(f"  {p['name']}: {p['count']}/{p['daily_limit']} ({pct}%)")
        return "\n".join(lines) or "  Нет доступных провайдеров"


rotator = ProviderRotator()

# ============================================================
#  ПОИСК МЕСТ — 2GIS (бесплатно) + OpenStreetMap (резерв)
# ============================================================

# Категории с тегами для 2GIS и OSM
PLACE_CATEGORIES = {
    "restaurant": {
        "triggers": [
            "ресторан", "рестораны", "поужинать", "пообедать", "поесть",
            "покушать", "кафе", "столовая", "бистро", "закусочная",
            "перекусить", "фастфуд", "суши", "пицца", "шаурма",
        ],
        "dgis_q": "кафе ресторан",
        "osm_tags": ['"amenity"="restaurant"', '"amenity"="cafe"',
                     '"amenity"="fast_food"', '"amenity"="bistro"'],
        "emoji": "🍽️",
        "label": "Рестораны и кафе",
    },
    "breakfast": {
        "triggers": ["позавтракать", "завтрак", "кофейня", "кофе"],
        "dgis_q": "кофейня завтрак",
        "osm_tags": ['"amenity"="cafe"', '"amenity"="coffee_shop"'],
        "emoji": "☕",
        "label": "Кофейни и завтрак",
    },
    "lunch": {
        "triggers": ["обед", "бизнес-ланч", "ланч", "столовая"],
        "dgis_q": "столовая бизнес-ланч",
        "osm_tags": ['"amenity"="restaurant"', '"amenity"="canteen"'],
        "emoji": "🥗",
        "label": "Обеды и ланчи",
    },
    "dinner": {
        "triggers": ["ужин", "поужинать", "вечером поесть"],
        "dgis_q": "ресторан ужин",
        "osm_tags": ['"amenity"="restaurant"'],
        "emoji": "🍷",
        "label": "Ужин и рестораны",
    },
    "hospital": {
        "triggers": [
            "больница", "больницы", "госпиталь", "скорая",
            "поликлиника", "поликлиники", "клиника", "медцентр",
            "медицинская", "врач", "врачи", "лечение",
        ],
        "dgis_q": "больница поликлиника клиника",
        "osm_tags": ['"amenity"="hospital"', '"amenity"="clinic"'],
        "emoji": "🏥",
        "label": "Больницы и поликлиники",
    },
    "pharmacy": {
        "triggers": ["аптека", "аптеки", "лекарства", "лекарство", "таблетки"],
        "dgis_q": "аптека",
        "osm_tags": ['"amenity"="pharmacy"'],
        "emoji": "💊",
        "label": "Аптеки",
    },
    "school": {
        "triggers": ["школа", "школы", "общеобразовательная", "гимназия", "лицей"],
        "dgis_q": "школа гимназия лицей",
        "osm_tags": ['"amenity"="school"'],
        "emoji": "🏫",
        "label": "Школы и гимназии",
    },
    "kindergarten": {
        "triggers": [
            "детский сад", "детские сады", "садик", "садики",
            "дошкольное", "детсад", "ясли",
        ],
        "dgis_q": "детский сад",
        "osm_tags": ['"amenity"="kindergarten"'],
        "emoji": "🎒",
        "label": "Детские сады",
    },
    "supermarket": {
        "triggers": [
            "магазин", "магазины", "супермаркет", "продукты",
            "продуктовый", "гипермаркет", "торговый",
        ],
        "dgis_q": "супермаркет магазин продукты",
        "osm_tags": ['"shop"="supermarket"', '"shop"="convenience"'],
        "emoji": "🛒",
        "label": "Магазины и супермаркеты",
    },
    "bank": {
        "triggers": ["банк", "банки", "банкомат", "снять деньги", "банкоматы"],
        "dgis_q": "банк банкомат",
        "osm_tags": ['"amenity"="bank"', '"amenity"="atm"'],
        "emoji": "🏦",
        "label": "Банки и банкоматы",
    },
    "gas_station": {
        "triggers": ["заправка", "заправки", "азс", "бензин", "топливо", "дизель"],
        "dgis_q": "АЗС заправка",
        "osm_tags": ['"amenity"="fuel"'],
        "emoji": "⛽",
        "label": "Заправки (АЗС)",
    },
    "hotel": {
        "triggers": [
            "отель", "отели", "гостиница", "гостиницы",
            "переночевать", "хостел", "апартаменты",
        ],
        "dgis_q": "отель гостиница хостел",
        "osm_tags": ['"tourism"="hotel"', '"tourism"="hostel"'],
        "emoji": "🏨",
        "label": "Отели и гостиницы",
    },
    "park": {
        "triggers": ["парк", "парки", "сквер", "погулять", "прогуляться", "зелёная зона"],
        "dgis_q": "парк сквер",
        "osm_tags": ['"leisure"="park"'],
        "emoji": "🌳",
        "label": "Парки и скверы",
    },
    "gym": {
        "triggers": ["спортзал", "фитнес", "тренажёрный зал", "спорт", "качалка"],
        "dgis_q": "спортзал фитнес",
        "osm_tags": ['"leisure"="fitness_centre"', '"leisure"="sports_centre"'],
        "emoji": "💪",
        "label": "Спортзалы и фитнес",
    },
    "beauty": {
        "triggers": ["салон красоты", "парикмахерская", "барбершоп", "маникюр", "стрижка"],
        "dgis_q": "салон красоты парикмахерская",
        "osm_tags": ['"shop"="hairdresser"', '"shop"="beauty"'],
        "emoji": "💅",
        "label": "Салоны красоты",
    },
}

# Собираем все триггеры в один список
ALL_PLACE_TRIGGERS: list = []
for _cat in PLACE_CATEGORIES.values():
    ALL_PLACE_TRIGGERS.extend(_cat["triggers"])


def extract_city_from_text(text: str):
    """Извлекаем название города из текста запроса."""
    stopwords = {
        "мая", "привет", "пожалуйста", "скажи", "расскажи", "найди", "покажи",
        "какая", "какое", "какой", "какие", "сейчас", "сегодня", "завтра",
        "время", "погода", "погоду", "температура", "прогноз",
        "который", "сколько", "часов", "часовой", "пояс",
        "ресторан", "рестораны", "кафе", "больница", "школа", "аптека",
        "магазин", "банк", "отель", "парк", "заправка", "садик",
        "детский", "сад", "поликлиника", "клиника", "гостиница",
        "где", "есть", "можно", "адрес", "находится", "рядом", "открыт",
        "работает", "хочу", "нужна", "нужно", "посоветуй", "порекомендуй",
    }
    # Ищем слова с заглавной буквы — это потенциальные названия городов
    words = re.findall(r'[А-ЯЁA-Z][а-яёa-z]{2,}', text)
    for word in words:
        if word.lower() not in stopwords:
            return word
    # Если заглавных нет — ищем после предлогов "в", "во"
    match = re.search(r'\bв\s+([а-яёА-ЯЁ]{3,})', text, re.IGNORECASE)
    if match:
        candidate = match.group(1)
        if candidate.lower() not in stopwords:
            return candidate.capitalize()
    return None


def detect_place_query(text: str):
    """
    Определяем тип поиска мест и город.
    Возвращает (category_key, city) или (None, None).
    """
    text_lower = text.lower()
    if not any(tr in text_lower for tr in ALL_PLACE_TRIGGERS):
        return None, None

    detected = None
    for cat_key, cat in PLACE_CATEGORIES.items():
        if any(tr in text_lower for tr in cat["triggers"]):
            detected = cat_key
            break

    city = extract_city_from_text(text)
    return detected, city


# ── 2GIS API (бесплатно, без карты) ─────────────────────────

def search_places_2gis(query: str, city: str, max_results: int = 8) -> list:
    """
    Поиск через 2GIS Catalog API.
    Регистрация: https://dev.2gis.com — бесплатно, без карты.
    Возвращает список заведений с адресами и телефонами.
    """
    if not DGIS_KEY:
        return []
    try:
        # Геокодируем город чтобы получить его ID для 2GIS
        geo_url = "https://catalog.api.2gis.com/3.0/items/geocode"
        geo_r = requests.get(geo_url, params={
            "q": city,
            "key": DGIS_KEY,
            "fields": "items.point",
        }, timeout=10)
        geo_data = geo_r.json()

        point = None
        items = geo_data.get("result", {}).get("items", [])
        if items:
            point_data = items[0].get("point", {})
            point = f"{point_data.get('lon')},{point_data.get('lat')}"

        # Поиск заведений
        search_url = "https://catalog.api.2gis.com/3.0/items"
        params = {
            "q": f"{query} {city}",
            "key": DGIS_KEY,
            "page_size": max_results,
            "fields": (
                "items.address,items.contact_groups,"
                "items.schedule,items.rubrics,"
                "items.rating,items.reviews_count"
            ),
            "locale": "ru_RU",
        }
        if point:
            params["point"] = point
            params["radius"] = 10000  # 10 км от центра

        r = requests.get(search_url, params=params, timeout=12)
        data = r.json()

        result = data.get("result", {})
        raw_items = result.get("items", [])

        places = []
        for item in raw_items[:max_results]:
            name = item.get("name", "Без названия")

            # Адрес
            address_obj = item.get("address", {})
            address = address_obj.get("name", "")
            if not address:
                components = address_obj.get("components", [])
                address = ", ".join(c.get("street_name", "") + " " + c.get("building", "")
                                    for c in components if c.get("street_name")).strip()

            # Телефоны
            phone = ""
            contacts = item.get("contact_groups", [])
            for group in contacts:
                for contact in group.get("contacts", []):
                    if contact.get("type") == "phone":
                        phone = contact.get("value", "")
                        break
                if phone:
                    break

            # Сайт
            website = ""
            for group in contacts:
                for contact in group.get("contacts", []):
                    if contact.get("type") in ("website", "url"):
                        website = contact.get("value", "")
                        break
                if website:
                    break

            # Режим работы — сегодня
            schedule = item.get("schedule", {})
            now_utc = datetime.now(timezone.utc)
            weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            today_key = weekdays[now_utc.weekday()]
            today_schedule = schedule.get(today_key, {})
            hours_str = ""
            if today_schedule:
                working = today_schedule.get("working_hours", [])
                if working:
                    from_t = working[0].get("from", "")
                    to_t   = working[0].get("to", "")
                    if from_t and to_t:
                        hours_str = f"{from_t}–{to_t}"
            is_24h = schedule.get("is_24hours", False)
            if is_24h:
                hours_str = "Круглосуточно"

            # Рейтинг
            rating = item.get("rating", {}).get("value") if item.get("rating") else None

            places.append({
                "name": name,
                "address": address or "Адрес не указан",
                "phone": phone,
                "website": website,
                "hours": hours_str,
                "rating": rating,
                "open_now": None,
                "source": "2GIS",
            })

        print(f"[2gis] Найдено {len(places)} для '{query}' в '{city}'")
        return places

    except Exception as e:
        print(f"[2gis error] {e}")
        return []


# ── OpenStreetMap / Overpass (резерв, без ключа) ─────────────

def search_places_osm(category_key: str, city: str, max_results: int = 8) -> list:
    """
    Поиск через Overpass API (OpenStreetMap).
    Полностью бесплатно, без ключа. Резерв при отсутствии 2GIS ключа.
    """
    cat = PLACE_CATEGORIES.get(category_key, {})
    osm_tags = cat.get("osm_tags", [])
    if not osm_tags or not city:
        return []
    try:
        # Геокодируем через Nominatim
        geo_r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": "MayaAI-VK-Bot/2.0"},
            timeout=10
        )
        geo_data = geo_r.json()
        if not geo_data:
            print(f"[osm] Город '{city}' не найден")
            return []

        lat = float(geo_data[0]["lat"])
        lon = float(geo_data[0]["lon"])
        radius = 5000  # 5 км

        tag_queries = "".join(
            f'node[{t}](around:{radius},{lat},{lon});'
            f'way[{t}](around:{radius},{lat},{lon});'
            for t in osm_tags
        )
        query = f"[out:json][timeout:20];({tag_queries});out center {max_results * 4};"

        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers={"User-Agent": "MayaAI-VK-Bot/2.0"},
            timeout=25
        )
        elements = r.json().get("elements", [])

        places = []
        seen = set()
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name") or tags.get("name:ru")
            if not name or name in seen:
                continue
            seen.add(name)

            # Строим адрес из тегов OSM
            parts = []
            if tags.get("addr:street"):
                s = tags["addr:street"]
                if tags.get("addr:housenumber"):
                    s += ", " + tags["addr:housenumber"]
                parts.append(s)
            city_tag = tags.get("addr:city") or tags.get("addr:town") or city
            if city_tag:
                parts.append(city_tag)

            places.append({
                "name": name,
                "address": ", ".join(parts) if parts else "Адрес не указан в OSM",
                "phone": tags.get("phone") or tags.get("contact:phone", ""),
                "website": tags.get("website") or tags.get("contact:website", ""),
                "hours": tags.get("opening_hours", ""),
                "rating": None,
                "open_now": None,
                "source": "OpenStreetMap",
            })
            if len(places) >= max_results:
                break

        print(f"[osm] Найдено {len(places)} для '{category_key}' в '{city}'")
        return places
    except Exception as e:
        print(f"[osm error] {e}")
        return []


def format_places(places: list, category_key: str, city: str) -> str:
    """Форматируем список мест в красивый текст."""
    if not places:
        return ""
    cat = PLACE_CATEGORIES.get(category_key, {})
    emoji = cat.get("emoji", "📍")
    label = cat.get("label", "Места")
    source = places[0].get("source", "")

    lines = [f"{emoji} {label} — {city}:\n"]
    for i, p in enumerate(places, 1):
        # Название и рейтинг
        line = f"{i}. {p['name']}"
        if p.get("rating"):
            line += f" ⭐ {p['rating']}"
        lines.append(line)

        # Адрес
        addr = p.get("address", "")
        if addr and addr not in ("Адрес не указан", "Адрес не указан в OSM"):
            lines.append(f"   📍 {addr}")

        # Статус работы
        if p.get("open_now") is True:
            lines.append("   ✅ Сейчас открыто")
        elif p.get("open_now") is False:
            lines.append("   ❌ Сейчас закрыто")

        # Часы
        if p.get("hours"):
            lines.append(f"   🕐 Сегодня: {p['hours']}")

        # Телефон
        if p.get("phone"):
            lines.append(f"   📞 {p['phone']}")

        # Сайт
        if p.get("website"):
            lines.append(f"   🌐 {p['website']}")

        lines.append("")

    lines.append(f"_Источник данных: {source}_")
    return "\n".join(lines)


def search_places(category_key: str, city: str):
    """
    Главная функция поиска:
    1. Пробуем 2GIS (бесплатно, точные данные)
    2. Фолбэк на OpenStreetMap (без ключа)
    """
    cat = PLACE_CATEGORIES.get(category_key)
    if not cat or not city:
        return None

    # 2GIS — приоритет
    if DGIS_KEY:
        places = search_places_2gis(cat["dgis_q"], city, max_results=8)
        if places:
            return format_places(places, category_key, city)

    # OSM — резерв
    places = search_places_osm(category_key, city, max_results=8)
    if places:
        return format_places(places, category_key, city)

    return None

# ============================================================
#  ПОГОДА И ВРЕМЯ
# ============================================================
WEATHER_TRIGGERS = [
    "погода", "погоду", "температура", "температуру",
    "дождь", "снег", "ясно", "облачно", "ветер",
    "градус", "климат", "прогноз", "weather",
]
TIME_TRIGGERS = [
    "время", "времени", "сколько времени", "который час",
    "часовой пояс", "timezone", "время в",
]

def has_weather_intent(text: str) -> bool:
    return any(kw in text.lower() for kw in WEATHER_TRIGGERS)

def has_time_intent(text: str) -> bool:
    return any(kw in text.lower() for kw in TIME_TRIGGERS)

def fetch_weather(city: str):
    try:
        r = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": WEATHER_API_KEY, "units": "metric", "lang": "ru"},
            timeout=10
        )
        d = r.json()
        if d.get("cod") != 200:
            return None
        ofs = d.get("timezone", 0)
        ldt = datetime.now(timezone.utc) + timedelta(seconds=ofs)
        return (
            f"📍 {d['name']}, {d['sys']['country']}\n"
            f"🕐 Местное время: {ldt.strftime('%H:%M, %d.%m.%Y')}\n"
            f"🌡️ {d['main']['temp']:.1f}°C (ощущается {d['main']['feels_like']:.1f}°C)\n"
            f"☁️ {d['weather'][0]['description'].capitalize()}\n"
            f"💧 Влажность: {d['main']['humidity']}%\n"
            f"💨 Ветер: {d['wind']['speed']} м/с"
        )
    except Exception as e:
        print(f"[weather error] {e}")
        return None

def fetch_time_only(city: str):
    try:
        r = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": WEATHER_API_KEY, "units": "metric", "lang": "ru"},
            timeout=10
        )
        d = r.json()
        if d.get("cod") != 200:
            return None
        ofs = d.get("timezone", 0)
        ldt = datetime.now(timezone.utc) + timedelta(seconds=ofs)
        return f"🕐 Текущее время в {d['name']}, {d['sys']['country']}: {ldt.strftime('%H:%M, %d.%m.%Y')}"
    except Exception as e:
        print(f"[time error] {e}")
        return None

def get_weather_context(text: str):
    city = extract_city_from_text(text)
    if not city:
        return None
    return fetch_weather(city)

def get_time_context(text: str):
    city = extract_city_from_text(text)
    if not city:
        return None
    return fetch_time_only(city)
    
# ============================================================
#  ПРОВЕРКА ФАКТОВ ЧЕРЕЗ ВИКИПЕДИЮ
# ============================================================
FACT_CHECK_TRIGGERS = [
    "кто", "что такое", "когда", "сколько", "где находится", "самый",
    "самая", "самое", "какой год", "столица", "население", "рекорд",
    "чемпион", "президент", "основан", "изобрёл", "изобрел", "год основания",
]

def looks_like_factual_query(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in FACT_CHECK_TRIGGERS)

def fetch_wikipedia_context(query: str, lang: str = "ru"):
    try:
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        search_r = requests.get(api_url, params={
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": 1,
        }, timeout=8, headers={"User-Agent": "MayaAI-VK-Bot/2.0"})
        results = search_r.json().get("query", {}).get("search", [])
        if not results:
            return None
        title = results[0]["title"]

        extract_r = requests.get(api_url, params={
            "action": "query", "prop": "extracts", "exintro": True,
            "explaintext": True, "titles": title, "format": "json",
        }, timeout=8, headers={"User-Agent": "MayaAI-VK-Bot/2.0"})
        pages = extract_r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                return f"[Википедия, статья «{title}»]: {extract[:1200]}"
        return None
    except Exception as e:
        print(f"[wiki error] {e}")
        return None

def get_fact_check_context(text: str):
    ctx = fetch_wikipedia_context(text, lang="ru")
    if not ctx:
        ctx = fetch_wikipedia_context(text, lang="en")
    return ctx
    
# ============================================================
#  ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ
# ============================================================
IMAGE_TRIGGERS = [
    "нарисуй", "нарисовать", "сгенерируй", "сгенерировать",
    "создай изображение", "создай картинку", "создай рисунок",
    "сделай картинку", "изобрази", "создай арт",
]

def is_image_request(text: str) -> bool:
    return any(kw in text.lower() for kw in IMAGE_TRIGGERS)

def extract_image_prompt(text: str) -> str:
    t = text.lower()
    for kw in sorted(IMAGE_TRIGGERS, key=len, reverse=True):
        t = t.replace(kw, "")
    for f in ["мне", "пожалуйста", "пожалуйста,"]:
        t = t.replace(f, "")
    return t.strip(" ,.:!?")

def translate_prompt_to_english(prompt: str) -> str:
    if not GROQ_KEY:
        return prompt
    try:
        r = requests.post(
            GROQ_URL,
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                   {"role": "system", "content": (
                        "Translate the image description to English for AI image generation. "
                        "Return ONLY the translated prompt with quality tags: "
                        "highly detailed, sharp focus, professional lighting, 4k, "
                        "photorealistic, coherent anatomy, correct proportions, "
                        "no distortion, no extra limbs, no blurring, clean composition."
                    )},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 200, "temperature": 0.3,
            },
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            timeout=15
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[translate error] {e}")
    return prompt

def generate_via_huggingface(prompt: str):
    if not HF_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    for model in [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ]:
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "width": 1024, "height": 1024,
                    "num_inference_steps": 4 if "schnell" in model else 30,
                    "guidance_scale": 0.0 if "schnell" in model else 7.5,
                },
            }
            r = requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                json=payload, headers=headers, timeout=120
            )
            ct = r.headers.get("content-type", "")
            if r.status_code == 200 and ct.startswith("image"):
                print(f"[image] ✅ HF: {model}")
                return r.content
            if r.status_code == 503:
                time.sleep(20)
                r2 = requests.post(
                    f"https://api-inference.huggingface.co/models/{model}",
                    json=payload, headers=headers, timeout=120
                )
                if r2.status_code == 200 and r2.headers.get("content-type", "").startswith("image"):
                    return r2.content
        except Exception as e:
            print(f"[image] HF ({model}): {e}")
    return None

def generate_via_together_flux(prompt: str):
    if not TOGETHER_KEY:
        return None
    try:
        r = requests.post(
            "https://api.together.xyz/v1/images/generations",
            json={
                "model": "black-forest-labs/FLUX.1-schnell-Free",
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "n": 1,
                "response_format": "base64",
            },
            headers={"Authorization": f"Bearer {TOGETHER_KEY}", "Content-Type": "application/json"},
            timeout=60
        )
        if r.status_code == 200:
            b64 = r.json()["data"][0]["b64_json"]
            print("[image] ✅ Together FLUX")
            return base64.b64decode(b64)
        print(f"[image] Together FLUX {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[image] Together FLUX error: {e}")
    return None

def generate_via_pollinations(prompt: str):
    try:
        encoded = requests.utils.quote(prompt)
        r = requests.get(
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width=1024&height=1024&nologo=true&enhance=true"
            f"&model=flux&seed={int(time.time())}",
            timeout=120
        )
        if r.status_code == 200 and len(r.content) > 1000:
            print("[image] ✅ Pollinations")
            return r.content
    except Exception as e:
        print(f"[image] Pollinations: {e}")
    return None

def generate_image(prompt: str):
    eng = translate_prompt_to_english(prompt)
    return generate_via_together_flux(eng) or generate_via_pollinations(eng)

# ============================================================
#  ГРАФИКИ ФУНКЦИЙ
# ============================================================
GRAPH_TRIGGERS = [
    "график функции", "построй график", "нарисуй график",
    "график уравнения", "начерти график", "изобрази график",
]

def is_graph_request(text: str) -> bool:
    return any(kw in text.lower() for kw in GRAPH_TRIGGERS)

def extract_function_expression(text: str):
    messages = [
        {"role": "system", "content": (
            "Извлеки из запроса пользователя математическую функцию от x "
            "в виде чистого выражения для sympy (пример: x**2 + 3*x - 5, "
            "sin(x), sqrt(x), log(x), exp(x)). "
            "Ответь ТОЛЬКО самим выражением, без слов, без 'y=', без пояснений."
        )},
        {"role": "user", "content": text},
    ]
    try:
        return call_ai_with_rotation(messages).strip()
    except Exception as e:
        print(f"[graph extract error] {e}")
        return None

def plot_function(expr_str: str):
    try:
        x = sp.symbols('x')
        expr = sp.sympify(expr_str, locals={"x": x})
        f = sp.lambdify(x, expr, modules=["numpy"])

        xs = np.linspace(-10, 10, 1000)
        with np.errstate(all="ignore"):
            ys = np.array(f(xs), dtype=float)
        ys[np.abs(ys) > 1e4] = np.nan

        plt.figure(figsize=(8, 6), dpi=150)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.plot(xs, ys, color="#4A90D9", linewidth=2)
        plt.title(f"y = {expr_str}")
        plt.xlabel("x")
        plt.ylabel("y")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"[graph plot error] {e}")
        return None

# ============================================================
#  ФОТО VK
# ============================================================
def upload_image_to_vk(vk, peer_id: int, image_bytes: bytes):
    try:
        server = vk.photos.getMessagesUploadServer(peer_id=peer_id)
        resp = requests.post(
            server["upload_url"],
            files={"photo": ("image.png", image_bytes, "image/png")},
            timeout=30
        )
        data = resp.json()
        saved = vk.photos.saveMessagesPhoto(
            photo=data["photo"], server=data["server"], hash=data["hash"]
        )
        return f"photo{saved[0]['owner_id']}_{saved[0]['id']}"
    except Exception as e:
        print(f"[vk upload error] {e}")
        return None

def download_photo_from_attachments(attachments: list):
    try:
        for att in attachments:
            if att.get("type") == "photo":
                sizes = att["photo"].get("sizes", [])
                if sizes:
                    best = max(sizes, key=lambda s: s.get("width", 0))
                    url = best.get("url")
                    if url:
                        return requests.get(url, timeout=30).content
    except Exception as e:
        print(f"[photo dl] {e}")
    return None

# ============================================================
#  ПОДПИСКА VK DONUT
# ============================================================
def get_user_subscription_level(vk, user_id: int) -> str:
    try:
        if not vk.donut.isDon(owner_id=-GROUP_ID, user_id=user_id):
            return "free"
        subs = vk.donut.getSubscription(owner_id=-GROUP_ID, user_id=user_id)
        amount = subs.get("amount", 0)
        level = "basic"
        for price, lvl in sorted(DONUT_LEVELS.items(), reverse=True):
            if amount >= price:
                level = lvl
                break
        return level
    except Exception:
        return "free"

# ============================================================
#  ИНДИКАТОР "ПЕЧАТАЕТ"
# ============================================================
def start_typing(vk, chat_peer_id: int, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            vk.messages.setActivity(peer_id=chat_peer_id, type="typing", group_id=GROUP_ID)
        except Exception:
            pass
        stop_event.wait(5)

def with_typing(vk, chat_peer_id: int, func, *args, **kwargs):
    stop_event = threading.Event()
    t = threading.Thread(target=start_typing, args=(vk, chat_peer_id, stop_event), daemon=True)
    t.start()
    try:
        return func(*args, **kwargs)
    finally:
        stop_event.set()

# ============================================================
#  ЗАПРОС К ИИ
# ============================================================
def call_ai_with_rotation(messages: list) -> str:
    last_error = None
    for _ in range(len(rotator.providers) + 1):
        provider = rotator.get_provider()
        if provider is None:
            print("[rotator] Все исчерпаны. Жду 60 сек...")
            time.sleep(60)
            provider = rotator.get_provider()
            if provider is None:
                break

        headers = {"Authorization": f"Bearer {provider['key']}", "Content-Type": "application/json"}
        if "openrouter" in provider["url"]:
            headers.update({"HTTP-Referer": "https://vk.com", "X-Title": "MayaAI"})

        try:
            r = requests.post(
                provider["url"],
                json={"model": provider["model"], "messages": messages,
                      "max_tokens": 1500, "temperature": 0.4},
                headers=headers, timeout=60
            )
            if r.status_code == 429:
                rotator.mark_error("429"); time.sleep(2); continue
            if r.status_code != 200:
                print(f"[ai] {provider['name']} {r.status_code}")
                rotator.mark_error(str(r.status_code)); continue
            content = (r.json().get("choices", [{}])[0]
                       .get("message", {}).get("content"))
            if not content or not content.strip():
                raise ValueError("Пустой ответ")
            rotator.increment()
            print(f"[ai] ✅ {provider['name']}")
            return content.strip()
        except Exception as e:
            last_error = str(e)
            print(f"[ai error | {provider['name']}] {e}")
            rotator.mark_error(last_error)
            time.sleep(1)

    raise RuntimeError(f"Все провайдеры недоступны: {last_error}")


def call_ai_premium(messages: list, level: str) -> str:
    cfg = PREMIUM_MODELS.get(level, PREMIUM_MODELS["standard"])
    r = requests.post(
        cfg["url"],
        json={"model": cfg["model"], "messages": messages,
              "max_tokens": 1500, "temperature": 0.4},
        headers={
            "Authorization": f"Bearer {cfg['key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://vk.com",
            "X-Title": "MayaAI",
        },
        timeout=60
    )
    r.raise_for_status()
    content = r.json().get("choices", [{}])[0].get("message", {}).get("content")
    if not content:
        raise ValueError("Пустой ответ премиум")
    return content.strip()


def ask_maya(
    text: str,
    peer_id: int,
    user_id: int,
    image_bytes=None,
    extra_context: str = None,
    sub_level: str = "free",
) -> str:
    history = get_history(peer_id)
    speaker_tag = f"[Пользователь {user_id}]: "

    if image_bytes:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            fmt = (img.format or "jpeg").lower()
        except Exception:
            fmt = "jpeg"
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        user_content = [{"type": "text", "text": speaker_tag + (text or "")}]
        if extra_context:
            user_content.append({"type": "text", "text": f"[Данные]: {extra_context}"})
        user_content.append({"type": "image_url",
                              "image_url": {"url": f"data:image/{fmt};base64,{b64}"}})
        use_vision = True
    else:
        msg = speaker_tag + text
        if extra_context:
            msg += f"\n\n[Актуальные данные]:\n{extra_context}"
        user_content = msg
        use_vision = False

    WEEKDAYS_RU = ["понедельник", "вторник", "среда", "четверг",
                   "пятница", "суббота", "воскресенье"]
    now_msk = datetime.now(timezone.utc) + timedelta(hours=3)
    weekday_name = WEEKDAYS_RU[now_msk.weekday()]
    current_date_note = (
        f"Сейчас точно: {weekday_name}, {now_msk.strftime('%d.%m.%Y')}, "
        f"время {now_msk.strftime('%H:%M')} (МСК). Это данные из системных часов сервера — "
        f"стопроцентно верные. Никогда не пересчитывай день недели самостоятельно."
    )
    group_chat_note = (
        "Внимание: это может быть групповой чат, где пишут разные пользователи. "
        "Каждое сообщение помечено как [Пользователь ID]. Учитывай, кто именно "
        "задаёт текущий вопрос — если он продолжает тему, поднятую другим пользователем, "
        "отвечай по существу этой темы, опираясь на предыдущие сообщения в истории, "
        "даже если их писал не он."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": current_date_note},
        {"role": "system", "content": group_chat_note},
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    try:
        if sub_level in ("standard", "premium"):
            answer = call_ai_premium(messages, sub_level)
        elif use_vision:
            answer = call_ai_premium(messages, "standard")
        else:
            answer = call_ai_with_rotation(messages)
        save_to_history(peer_id, "user", user_content)
        save_to_history(peer_id, "assistant", answer)
        return answer
    except Exception as e:
        print(f"[ask_maya error] {e}")
        try:
            answer = call_ai_with_rotation(messages)
            save_to_history(peer_id, "user", user_content)
            save_to_history(peer_id, "assistant", answer)
            return answer
        except Exception as e2:
            print(f"[ask_maya fallback error] {e2}")

    return "Произошла временная ошибка. Попробуй снова через несколько секунд."

# ============================================================
#  ОТПРАВКА С ЦИТИРОВАНИЕМ
# ============================================================
def send_message(vk, peer_id: int, text: str,
                 attachment: str = None, conv_message_id: int = None):
    kwargs = {"peer_id": peer_id, "message": text,
              "random_id": int(time.time() * 10000)}
    if attachment:
        kwargs["attachment"] = attachment
    if conv_message_id:
        kwargs["forward"] = json.dumps({
            "peer_id": peer_id,
            "conversation_message_ids": [conv_message_id],
            "is_reply": 1,
        }, ensure_ascii=False)
    try:
        vk.messages.send(**kwargs)
    except Exception as e:
        print(f"[send forward error] {e}")
        try:
            kwargs.pop("forward", None)
            kwargs["random_id"] = int(time.time() * 10000) + 1
            vk.messages.send(**kwargs)
        except Exception as e2:
            print(f"[send error] {e2}")

# ============================================================
#  НУЖНО ЛИ ОТВЕЧАТЬ В БЕСЕДЕ
# ============================================================
def should_respond_in_chat(message: dict) -> tuple:
    text = (message.get("text") or "").strip()
    peer_id = message.get("peer_id", 0)

    if peer_id <= 2000000000:
        return True, text

    mention_re = re.compile(rf'\[club{GROUP_ID}\|[^\]]*\]', re.IGNORECASE)
    if mention_re.search(text):
        clean = mention_re.sub("", text).strip()
        print(f"[chat] Упоминание → '{clean}'")
        return True, clean

    reply = message.get("reply_message")
    if reply and reply.get("from_id", 0) == -GROUP_ID:
        print(f"[chat] Reply на бота → отвечаю")
        return True, text

    return False, text

# ============================================================
#  ГЛАВНЫЙ ЦИКЛ
# ============================================================
def main():
    print("МАЯ запускается...")
    print(f"[config] GROUP_ID:       {GROUP_ID}")
    print(f"[config] VK_TOKEN:       {'✅' if VK_TOKEN else '❌'}")
    print(f"[config] SAMBANOVA_KEY:  {'✅' if SAMBANOVA_KEY else '❌'}")
    print(f"[config] GROQ_KEY:       {'✅' if GROQ_KEY else '❌'}")
    print(f"[config] CEREBRAS_KEY:   {'✅' if CEREBRAS_KEY else '❌'}")
    print(f"[config] TOGETHER_KEY:   {'✅' if TOGETHER_KEY else '❌'}")
    print(f"[config] FIREWORKS_KEY:  {'✅' if FIREWORKS_KEY else '❌'}")
    print(f"[config] MISTRAL_KEY:    {'✅' if MISTRAL_KEY else '❌'}")
    print(f"[config] OPENROUTER_KEY: {'✅' if OPENROUTER_KEY else '❌'}")
    print(f"[config] WEATHER_KEY:    {'✅' if WEATHER_API_KEY else '❌'}")
    print(f"[config] HF_TOKEN:       {'✅' if HF_TOKEN else '❌ (Pollinations)'}")
    print(f"[config] DGIS_KEY:       {'✅ 2GIS' if DGIS_KEY else '❌ (OpenStreetMap)'}")
    print(f"\n[rotator] Провайдеры:\n{rotator.status()}\n")

    vk_session = vk_api.VkApi(token=VK_TOKEN)
    vk = vk_session.get_api()
    group_info = vk.groups.getById(group_ids=GROUP_ID)[0]
    print(f"✅ Бот запущен: {group_info['name']} (id={GROUP_ID})")

    while True:
        try:
            longpoll = VkBotLongPoll(vk_session, GROUP_ID)
            for event in longpoll.listen():
                try:
                    if event.type != VkBotEventType.MESSAGE_NEW:
                        continue

                    message     = event.obj.message
                    peer_id     = message.get("peer_id", 0)
                    from_id     = message.get("from_id", 0)
                    attachments = message.get("attachments", [])
                    cmid        = message.get("conversation_message_id") or None

                    print(f"[event] peer={peer_id} from={from_id} cmid={cmid} "
                          f"text='{(message.get('text') or '')[:50]}'")

                    if from_id == -GROUP_ID:
                        continue

                    respond, text = should_respond_in_chat(message)
                    if not respond:
                        continue

                    sub_level = get_user_subscription_level(vk, from_id)
                    
                    # ── ГРАФИК ФУНКЦИИ ────────────────────────────
                    if text and is_graph_request(text):
                        expr = with_typing(vk, peer_id, extract_function_expression, text)
                        img = plot_function(expr) if expr else None
                        if img:
                            att = upload_image_to_vk(vk, peer_id, img)
                            send_message(vk, peer_id, f"Готово! График функции y = {expr}",
                                         attachment=att, conv_message_id=cmid)
                        else:
                            send_message(vk, peer_id,
                                         "Не удалось построить график. Уточни функцию, например: «построй график x^2 + 2x»",
                                         conv_message_id=cmid)
                        continue

                    # ── ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЯ ─────────────────────
                    if text and is_image_request(text):
                        prompt = extract_image_prompt(text)
                        if not prompt:
                            send_message(vk, peer_id, "Уточни, что именно нарисовать.",
                                         conv_message_id=cmid)
                            continue
                        send_message(vk, peer_id, "Генерирую изображение, несколько секунд…",
                                     conv_message_id=cmid)
                        img = with_typing(vk, peer_id, generate_image, prompt)
                        if img:
                            att = upload_image_to_vk(vk, peer_id, img)
                            send_message(vk, peer_id, f"Готово! Вот изображение по вашему запросу: «{text.strip()}»",
                                         attachment=att, conv_message_id=cmid)
                        else:
                            send_message(vk, peer_id,
                                         "Не удалось создать изображение. Попробуй другой запрос.",
                                         conv_message_id=cmid)
                        continue

                    # ── ФОТО ОТ ПОЛЬЗОВАТЕЛЯ ─────────────────────
                    image_bytes = download_photo_from_attachments(attachments)
                    if not text and image_bytes:
                        text = "Подробно опиши, что изображено на этом фото."
                    if not text and not image_bytes:
                        continue

                    # ── ПОИСК МЕСТ ────────────────────────────────
                    if text:
                        cat_key, city = detect_place_query(text)
                        if cat_key:
                            if not city:
                                send_message(
                                    vk, peer_id,
                                    "Уточни, пожалуйста, в каком городе или населённом пункте искать?",
                                    conv_message_id=cmid
                                )
                                continue

                            print(f"[places] {cat_key} → {city}")
                            places_ctx = with_typing(vk, peer_id, search_places, cat_key, city)

                            extra = places_ctx if places_ctx else (
                                f"По запросу «{text}» в {city} данные не найдены. "
                                f"Сообщи об этом вежливо, предложи уточнить запрос "
                                f"или проверить написание города."
                            )
                            answer = with_typing(
                                vk, peer_id, ask_maya,
                                text, peer_id, from_id, None, extra, sub_level
                            )
                            send_message(vk, peer_id, answer, conv_message_id=cmid)
                            continue

                    # ── ПОГОДА / ВРЕМЯ ────────────────────────────
                    extra_context = None
                    if text and has_weather_intent(text):
                        extra_context = get_weather_context(text)
                        if not extra_context:
                            send_message(vk, peer_id,
                                         "Не смогла определить город. "
                                         "Уточни запрос, например: «погода в Москве»",
                                         conv_message_id=cmid)
                            continue
                    elif text and has_time_intent(text):
                        extra_context = get_time_context(text)
                        if not extra_context:
                            send_message(vk, peer_id,
                                         "Не смогла определить город. "
                                         "Уточни запрос, например: «время в Токио»",
                                         conv_message_id=cmid)
                            continue

                    if text and not extra_context and looks_like_factual_query(text):
                        extra_context = with_typing(vk, peer_id, get_fact_check_context, text)

                    # ── ЗАПРОС К МАЯ ─────────────────────────────
                    answer = with_typing(
                        vk, peer_id, ask_maya,
                        text, peer_id, from_id, image_bytes, extra_context, sub_level
                    )
                    send_message(vk, peer_id, answer, conv_message_id=cmid)

                except Exception as e:
                    print(f"[event error] {e}")
                    time.sleep(1)

        except Exception as e:
            err = str(e)
            if any(x in err for x in ["ReadTimeout", "read timed out", "timed out"]):
                print("[longpoll] Таймаут — переподключаюсь...")
            elif any(x in err for x in ["ConnectionError", "ConnectionReset", "RemoteDisconnected"]):
                print("[longpoll] Потеря соединения — жду 5 сек...")
                time.sleep(5)
            else:
                print(f"[longpoll error] {e}")
                time.sleep(5)


if __name__ == "__main__":
    main()
