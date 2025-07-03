from gradio_client import Client
import os
from dotenv import load_dotenv
from PIL import Image
import requests
import logging

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

print(f"TELEGRAM_BOT_TOKEN: {TELEGRAM_BOT_TOKEN[:5]}... (скрыт)")
print(f"TELEGRAM_CHANNEL_ID: {TELEGRAM_CHANNEL_ID}")

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

# === Основной процесс ===
MODEL_NAME = get_working_model()
print(f"✅ Использую модель: {MODEL_NAME}")
client = Client(MODEL_NAME, hf_token=HF_TOKEN)
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    with open(INPUT_FILENAME, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"❌ Файл {INPUT_FILENAME} не найден.")
    exit()

def send_to_telegram(image_path, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as photo:
            files = {"photo": photo}
            data = {
                "chat_id": TELEGRAM_CHANNEL_ID,
                "caption": caption[:200],  # ограничение длины текста
                "parse_mode": "Markdown"
            }
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

for idx, prompt in enumerate(prompts):
    print(f"\n🔄 [{idx + 1}/{len(prompts)}] Генерация: «{prompt}»")
    try:
        # Параметры для FLUX.1-schnell
        if "FLUX.1-schnell" in MODEL_NAME:
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
        output_path = os.path.join(OUTPUT_DIR, f"{idx + 1}_{safe_prompt}.png")

        # Конвертация
        with Image.open(temp_image_path) as img:
            img.save(output_path, "PNG", quality=100)
        print(f"🖼️ Сохранено как PNG: {output_path}")
        logging.info(f"Image saved: {output_path}")

        # Отправка в Telegram
        if os.path.exists(output_path):
            print(f"📤 Отправляем в Telegram: {output_path}")
            success = send_to_telegram(output_path, caption=f"🖼️ {prompt}")
            if success:
                print("✅ Успешно отправлено в Telegram")
            else:
                print("❌ Не удалось отправить в Telegram")
        else:
            print(f"❌ Файл {output_path} не существует для отправки")

    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        logging.exception(f"Error processing prompt '{prompt}': {e}")

print("\n🎉 Обработка завершена!")