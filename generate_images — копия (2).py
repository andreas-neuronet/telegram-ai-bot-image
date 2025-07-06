from gradio_client import Client
import os
from dotenv import load_dotenv
from PIL import Image
import requests
import logging
from datetime import datetime

# === Настройки логирования ===
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Загрузка окружения ===
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
OUTPUT_DIR = "images"

# === Резервные модели ===
BACKUP_MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5"
]

def get_working_model():
    """Возвращает первую рабочую модель из списка"""
    model_list = []
    
    # Чтение моделей из файла
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "r", encoding="utf-8") as f:
                model_list = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"⚠️ Ошибка чтения {MODEL_FILE}: {e}")
    
    # Добавляем резервные модели
    model_list.extend(BACKUP_MODELS)
    
    # Проверяем доступность моделей
    for model in model_list:
        try:
            print(f"🔄 Проверка модели: {model}")
            Client(model, hf_token=HF_TOKEN)
            return model
        except Exception as e:
            print(f"❌ Модель {model} недоступна: {str(e)[:200]}")
            continue
    
    raise ValueError("❌ Ни одна из моделей не доступна!")

def load_prompts():
    """Загружает промпты из файла"""
    try:
        with open(INPUT_FILENAME, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Файл {INPUT_FILENAME} не найден")
        return []
    except Exception as e:
        print(f"❌ Ошибка чтения промптов: {e}")
        return []

def remove_first_prompt():
    """Удаляет первую строку из файла промптов"""
    try:
        # Получаем абсолютный путь
        input_path = os.path.abspath(INPUT_FILENAME)
        
        # Читаем текущее содержимое
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Если файл пуст
        if not lines:
            print("ℹ️ Файл промптов пуст")
            return True
        
        # Удаляем первую строку
        with open(input_path, "w", encoding="utf-8") as f:
            f.writelines(lines[1:])
        
        print(f"✅ Удален первый промпт. Осталось {len(lines)-1} строк.")
        return True
    
    except PermissionError:
        print("❌ Ошибка: Нет прав на запись в файл")
        return False
    except Exception as e:
        print(f"❌ Ошибка при обновлении файла: {e}")
        return False

def send_to_telegram(image_path):
    """Отправляет изображение в Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as photo:
            response = requests.post(
                url,
                files={"photo": photo},
                data={"chat_id": TELEGRAM_CHANNEL_ID}
            )
            
            if response.status_code == 200:
                print("✅ Изображение отправлено в Telegram")
                return True
            else:
                print(f"❌ Ошибка Telegram API: {response.status_code}")
                print(response.text[:200])
                return False
    except Exception as e:
        print(f"❌ Ошибка отправки в Telegram: {e}")
        return False

def generate_image(client, prompt, model_name):
    """Генерирует изображение по промпту"""
    try:
        # Параметры для разных моделей
        if "FLUX.1-schnell" in model_name:
            result = client.predict(
                prompt=prompt,
                api_name="/infer"
            )
        else:
            result = client.predict(
                prompt=prompt,
                seed=0,
                width=1024,
                height=1024,
                guidance_scale=3.5,
                num_inference_steps=28,
                api_name="/infer"
            )
        
        return result[0]  # Возвращаем временный путь к изображению
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
        return None

def save_image(temp_path, prompt):
    """Сохраняет изображение в постоянное хранилище"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Создаем безопасное имя файла
    safe_prompt = "".join(
        c for c in prompt[:30] 
        if c.isalnum() or c in " _-"
    ).rstrip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{safe_prompt}.png")
    
    try:
        with Image.open(temp_path) as img:
            img.save(output_path, "PNG", quality=95)
        print(f"🖼️ Изображение сохранено: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
        return None

def should_publish_now():
    """Проверяет, нужно ли публиковать сейчас"""
    now = datetime.now()
    return now.hour >= 19 or now.hour < 19  # С 19:00 до 18:00

def main():
    print("\n🚀 Запуск генерации изображений")
    
    if not should_publish_now():
        print("⏳ Сейчас не время публикации")
        return
    
    # Инициализация модели
    model_name = get_working_model()
    print(f"🔧 Используется модель: {model_name}")
    client = Client(model_name, hf_token=HF_TOKEN)
    
    # Загрузка промптов
    prompts = load_prompts()
    if not prompts:
        print("❌ Нет промптов для обработки")
        return
    
    prompt = prompts[0]
    print(f"\n🎨 Генерация: «{prompt}»")
    
    # Генерация изображения
    temp_image_path = generate_image(client, prompt, model_name)
    if not temp_image_path:
        return
    
    # Сохранение изображения
    final_image_path = save_image(temp_image_path, prompt)
    if not final_image_path:
        return
    
    # Отправка в Telegram
    if send_to_telegram(final_image_path):
        # Удаляем обработанный промпт только после успешной отправки
        remove_first_prompt()

if __name__ == "__main__":
    main()