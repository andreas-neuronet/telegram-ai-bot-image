from gradio_client import Client
import os
from dotenv import load_dotenv
from PIL import Image
import requests
import logging
from datetime import datetime

# === Настройки логирования ===
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# === Загрузка токена из .env ===
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

# Проверка обязательных переменных
if not HF_TOKEN:
    raise ValueError("❌ Не найден HF_TOKEN в .env файле.")
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHANNEL_ID:
    raise ValueError("❌ Не найдены TELEGRAM_BOT_TOKEN или TELEGRAM_CHANNEL_ID в .env файле.")

# === Настройки ===
MODEL_FILE = "MODEL.txt"
INPUT_FILENAME = "input.txt"
OUTPUT_DIR = "image"

# === Резервные модели ===
BACKUP_MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5"
]

def get_working_model():
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "r", encoding="utf-8") as f:
                models = [line.strip() for line in f if line.strip()]
                for model in models:
                    try:
                        print(f"🔄 Проверка модели: {model}")
                        Client(model, hf_token=HF_TOKEN)
                        return model
                    except Exception as e:
                        print(f"❌ Модель {model} недоступна: {e}")
                        continue
        except Exception as e:
            print(f"❌ Ошибка чтения файла моделей: {e}")

    print("⚠️ Использую резервные модели...")
    for model in BACKUP_MODELS:
        try:
            Client(model, hf_token=HF_TOKEN)
            return model
        except Exception as e:
            print(f"❌ Резервная модель {model} недоступна: {e}")
            continue

    raise ValueError("❌ Ни одна из моделей не доступна!")

def load_prompts():
    try:
        with open(INPUT_FILENAME, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
            return prompts
    except FileNotFoundError:
        print(f"❌ Файл {INPUT_FILENAME} не найден.")
        return []

def remove_first_prompt():
    try:
        # Читаем все строки из файла
        with open(INPUT_FILENAME, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Находим индекс первой непустой строки
        first_non_empty = None
        for i, line in enumerate(lines):
            if line.strip():  # Если строка не пустая
                first_non_empty = i
                break
        
        if first_non_empty is not None:
            # Удаляем первую непустую строку
            del lines[first_non_empty]
            
            # Перезаписываем файл
            with open(INPUT_FILENAME, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print("✅ Первый промпт успешно удален из файла.")
        else:
            print("ℹ️ Файл промптов пуст, нечего удалять.")
    except Exception as e:
        print(f"❌ Ошибка при обновлении файла промптов: {e}")
        logging.error(f"Error updating prompts file: {e}")

def send_to_telegram(image_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as photo:
            files = {"photo": photo}
            data = {"chat_id": TELEGRAM_CHANNEL_ID}
            response = requests.post(url, data=data, files=files)
            
            if response.status_code == 200:
                print("✅ Успешно отправлено в Telegram!")
                logging.info(f"Sent to Telegram: {image_path}")
                return True
            else:
                error_text = response.text
                print(f"❌ Ошибка при отправке в Telegram: код {response.status_code}")
                print("Ответ сервера:", error_text)
                logging.error(f"Telegram error: {error_text}")
                return False
    except Exception as e:
        print(f"⚠️ Исключение при отправке в Telegram: {e}")
        logging.exception("Telegram send error")
        return False

def generate_and_send_image():
    # Инициализация модели
    model_name = get_working_model()
    print(f"✅ Использую модель: {model_name}")
    client = Client(model_name, hf_token=HF_TOKEN)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prompts = load_prompts()
    if not prompts:
        print("❌ Нет доступных промптов для обработки.")
        return
    
    prompt = prompts[0]
    print(f"\n🔄 Генерация: «{prompt}»")
    try:
        # Параметры для FLUX.1-schnell
        if "FLUX.1-schnell" in model_name:
            result = client.predict(
                prompt=prompt,
                api_name="/infer"
            )
        else:  # Параметры для других моделей
            result = client.predict(
                prompt=prompt,
                seed=0,
                width=1024,
                height=1024,
                guidance_scale=3.5,
                num_inference_steps=28,
                api_name="/infer"
            )

        temp_image_path = result[0]
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " _-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{safe_prompt}.png")

        # Конвертация
        with Image.open(temp_image_path) as img:
            img.save(output_path, "PNG", quality=100)
        print(f"🖼️ Сохранено как PNG: {output_path}")
        logging.info(f"Image saved: {output_path}")

        # Отправка в Telegram
        if os.path.exists(output_path):
            print(f"📤 Отправляем в Telegram: {output_path}")
            success = send_to_telegram(output_path)
            if success:
                print("✅ Успешно отправлено в Telegram")
                remove_first_prompt()
            else:
                print("❌ Не удалось отправить в Telegram")
        else:
            print(f"❌ Файл {output_path} не существует для отправки")

    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        logging.exception(f"Error processing prompt '{prompt}': {e}")

def should_publish_now():
    now = datetime.now()
    current_hour = now.hour
    # Публиковать с 19:00 до 18:00 следующего дня (по МСК)
    return current_hour >= 19 or current_hour < 19

if __name__ == "__main__":
    print("\n🎉 Скрипт запущен. Проверка необходимости публикации...")
    if should_publish_now():
        generate_and_send_image()
    else:
        print("⏳ Сейчас не время публикации (работаем с 19:00 до 18:00)")