from gradio_client import Client
import os
from dotenv import load_dotenv
from PIL import Image

# === Загрузка токена из .env ===
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("❌ Не найден HF_TOKEN в .env файле.")

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

    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")

print("\n🎉 Обработка завершена!")

import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

def send_to_telegram(image_path, caption=""):
    url = f"https://api.telegram.org/bot {TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as photo:
        files = {"photo": photo}
        data = {"chat_id": TELEGRAM_CHANNEL_ID, "caption": caption}
        response = requests.post(url, data=data, files=files)
    return response.json()

# После сохранения
send_to_telegram(output_path, caption=f"🖼️ {prompt}")