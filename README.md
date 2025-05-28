# 🎨 Голостикеры - Telegram Bot для генерации стикеров

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/aiogram-3.7+-green.svg" alt="aiogram">
  <img src="https://img.shields.io/badge/Stable_Diffusion-1.5-purple.svg" alt="Stable Diffusion">
  <img src="https://img.shields.io/badge/Whisper-OpenAI-orange.svg" alt="Whisper">
</p>

Telegram-бот, который создает уникальные стикеры на основе голосовых сообщений или текстовых описаний с использованием OpenAI Whisper для распознавания речи и Stable Diffusion для генерации изображений.

## ✨ Возможности

- 🎤 **Голосовой ввод** - отправьте голосовое сообщение с описанием стикера
- ✍️ **Текстовый ввод** - или просто напишите текстовое описание
- 🎨 **Различные стили** - выберите стиль: обычный, мультяшный, аниме, минималистичный, эмодзи
- 🖼 **Настройка фона** - прозрачный, белый, градиент, круглый
- 📊 **Статистика** - отслеживайте количество созданных стикеров
- 🗂 **История** - просматривайте свои последние стикеры
- 🚀 **Оптимизация** - поддержка GPU (CUDA/MPS) для быстрой генерации

## 📋 Требования

- Python 3.10 или 3.11
- FFmpeg (для обработки аудио)
- 8+ ГБ RAM (16 ГБ рекомендуется)
- 10+ ГБ свободного места (для моделей)
- (Опционально) GPU с 6+ ГБ VRAM для ускорения

## 🚀 Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/yourusername/golostickery.git
cd golostickery
```

### 2. Создание виртуального окружения

```bash
# Для Python 3.11
python3.11 -m venv venv

# Активация (Linux/macOS)
source venv/bin/activate

# Активация (Windows)
venv\Scripts\activate
```

### 3. Установка FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Скачайте с [ffmpeg.org](https://ffmpeg.org/download.html) и добавьте в PATH.

### 4. Установка зависимостей

```bash
# Базовая установка
pip install -r requirements.txt

# Для Apple Silicon (M1/M2/M3)
pip install torch torchvision torchaudio
pip install openai-whisper

# Для CUDA (NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. Настройка бота

1. Создайте бота через [@BotFather](https://t.me/botfather)
2. Скопируйте `.env.example` в `.env`:
   ```bash
   cp .env.example .env
   ```
3. Откройте `.env` и вставьте токен вашего бота:
   ```
   BOT_TOKEN=your_bot_token_here
   ```

### 6. Запуск бота

```bash
python main.py
```

При первом запуске будут загружены модели (около 5-6 ГБ). Это займет некоторое время.

## 📁 Структура проекта

```
golostickery/
├── main.py                 # Точка входа
├── config.py              # Конфигурация
├── logger.py              # Настройка логирования
├── bot_handlers.py        # Обработчики сообщений
├── db_manager/
│   └── manager.py         # Работа с базой данных
├── stt_service/
│   └── service.py         # Распознавание речи (Whisper)
├── image_generation_service/
│   └── service.py         # Генерация изображений (SD)
├── sticker_utils/
│   └── utils.py           # Обработка стикеров
├── storage/               # Временные файлы
├── logs/                  # Логи
└── stickers.db           # База данных SQLite
```

## 🎮 Использование

1. **Начало работы**: Отправьте `/start` боту
2. **Создание стикера**:
   - Отправьте голосовое сообщение с описанием
   - Или напишите текстовое описание
3. **Выберите стиль**: Нажмите на кнопку с желаемым стилем
4. **Выберите фон**: Выберите тип фона для стикера
5. **Получите результат**: Бот отправит готовый стикер

### Команды бота

- `/start` - Начать работу с ботом
- `/help` - Показать справку
- `/stats` - Ваша статистика использования
- `/mystickers` - Показать последние 5 стикеров

## ⚙️ Конфигурация

### Переменные окружения (.env)

```env
# Обязательные
BOT_TOKEN=your_token_here

# Опциональные
WHISPER_MODEL=base              # tiny, base, small, medium, large
SD_DEVICE=mps                   # cpu, cuda, mps
SD_NUM_INFERENCE_STEPS=30       # 20-50 (меньше = быстрее)
```

### Оптимизация производительности

**Для Apple Silicon (M1/M2/M3):**
- Используйте `SD_DEVICE=mps` для ускорения на GPU
- Whisper автоматически использует оптимизации для Apple Silicon

**Для NVIDIA GPU:**
- Используйте `SD_DEVICE=cuda`
- Установите xformers для дополнительного ускорения:
  ```bash
  pip install xformers
  ```

**Для CPU:**
- Используйте `SD_NUM_INFERENCE_STEPS=20` для ускорения
- Рассмотрите модель Whisper `tiny` для быстрого распознавания

## 🛠 Разработка

### Запуск тестов

```bash
pytest tests/
```

### Форматирование кода

```bash
black .
flake8 .
```

### Pre-commit хуки

```bash
pre-commit install
pre-commit run --all-files
```

## 📊 Мониторинг

Бот автоматически создает логи в папке `logs/`:
- `bot.log` - основной лог
- `errors.log` - только ошибки

База данных хранит:
- Информацию о созданных стикерах
- Статистику пользователей
- Историю ошибок для анализа

## 🐛 Решение проблем

### "FFmpeg не найден"
Убедитесь, что FFmpeg установлен и доступен в PATH:
```bash
ffmpeg -version
```

### "Недостаточно памяти"
- Используйте меньшую модель Whisper (`tiny` или `base`)
- Уменьшите `SD_NUM_INFERENCE_STEPS`
- Включите CPU offloading (автоматически для CPU)

### "Генерация слишком медленная"
- Используйте GPU (CUDA или MPS)
- Уменьшите количество шагов генерации
- Используйте меньшее разрешение (уже оптимизировано до 512x512)

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE) для деталей.

## 🙏 Благодарности

- [OpenAI Whisper](https://github.com/openai/whisper) за распознавание речи
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) за генерацию изображений
- [aiogram](https://github.com/aiogram/aiogram) за отличный фреймворк для Telegram ботов

## 📞 Контакты

- Telegram: [@yourusername](https://t.me/yourusername)
- Email: your.email@example.com

---

<p align="center">Made with ❤️ and 🤖</p>