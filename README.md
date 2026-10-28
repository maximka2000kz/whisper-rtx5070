# Whisper STT для RTX 5070 (Blackwell, 12 ГБ) — Windows

Локальная транскрипция аудио и видео в текст. Оптимизировано под **RTX 5070**:
faster-whisper (CTranslate2), `large-v3-turbo`, батчевый инференс, VAD.

Ожидаемо на 5070: **2 часа видео ≈ 2–5 минут**.

---

## Самый простой путь: VS Code (Windows)

1. Распакуйте архив, в VS Code: **File → Open Folder…** → выберите папку `whisper-rtx5070`.
2. Установите расширение **Python** (Microsoft), если ещё нет.
3. Один раз выполните установку: откройте терминал (`Ctrl+~`) и запустите
   ```bat
   setup.bat
   ```
   (или `Terminal → Run Task… → Setup (.venv + dependencies)`).
4. `Ctrl+Shift+P` → **Python: Select Interpreter** → выберите
   `.venv\Scripts\python.exe`.
5. Нажмите **F5** → выберите **«Transcribe (faster-whisper)»** → введите путь
   к файлу или ссылку. Готовый текст появится в `transcript.txt`, субтитры — в `transcript.srt`.

Всё, больше ничего вводить не нужно. Если F5 не предлагает конфигурацию —
значит не открыта папка `whisper-rtx5070` (должна быть корнем workspace).

---

## Требования (Windows)

- **Windows 10/11 64-bit**
- **Драйвер NVIDIA 570+** — проверка: `nvidia-smi` в cmd
- **Python 3.10–3.14** с python.org, при установке галочка **Add Python to PATH**
- Интернет для первого скачивания модели (~1.6 ГБ)

CUDA Toolkit и cuDNN отдельно ставить **не нужно** — `setup.bat` поставит
runtime-DLL пакетами `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`.

---

## Установка

1. Распакуйте архив в любую папку (без кириллицы в пути — надёжнее).
2. Двойной клик по **`setup.bat`** (или в cmd):
   ```bat
   setup.bat
   ```
   С резервным движком (torch + transformers):
   ```bat
   setup.bat --with-transformers
   ```

Скрипт создаст `.venv`, поставит `faster-whisper`, `yt-dlp`, CUDA-DLL.

---

## Запуск

```bat
:: локальный файл (mp4/mkv/mp3/m4a/wav — звук берётся автоматически)
run.bat video.mp4 -o transcript.txt

:: со субтитрами
run.bat video.mp4 -o transcript.txt --srt transcript.srt

:: прямо по ссылке (Rutube, YouTube и т.п.)
run.bat "https://rutube.ru/video/<id>/" -o transcript.txt

:: другой язык
run.bat video.mp4 --lang en

:: резервный движок
run.bat video.mp4 --engine transformers
```

Без `run.bat` (вручную, из cmd):
```bat
.venv\Scripts\activate.bat
python transcribe.py video.mp4 -o transcript.txt
```

> Если `run.bat` закрывается мгновенно — запускайте из cmd, чтобы увидеть ошибку.

---

## Параметры

| Флаг | По умолчанию | Описание |
|---|---|---|
| `-o, --output` | `transcript.txt` | Файл с текстом |
| `--srt` | — | Дополнительно сохранить субтитры |
| `--engine` | `auto` | `auto` / `faster` / `transformers` |
| `--model` | `large-v3-turbo` | Модель faster-whisper |
| `--hf-model` | `openai/whisper-large-v3-turbo` | Модель для transformers |
| `--lang` | `ru` | Язык |
| `--task` | `transcribe` | `transcribe` или `translate` |
| `--batch` | `16` | Батч (на 12 ГБ можно 16–24) |
| `--beam` | `5` | Beam search |
| `--compute-type` | `float16` | `float16` / `int8_float16` / `int8` |

### Подбор под 12 ГБ

- **Максимум скорости:** `--compute-type int8_float16 --batch 24`
- **Максимум качества:** `--compute-type float16 --model large-v3 --batch 8`
- **Мало VRAM / ошибки:** `--compute-type int8 --batch 8`

---

## Проверка GPU

```bat
.venv\Scripts\activate.bat
python -c "import ctranslate2;print(ctranslate2.__version__, ctranslate2.get_cuda_device_count())"
```
Должно вывести версию и **1** (одна CUDA-карта).

Если ставили резервный движок:
```bat
python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

---

## Частые проблемы (Windows)

**`CUDA driver version is insufficient` / `no kernel image is available`**
Обновите драйвер NVIDIA, затем
```bat
.venv\Scripts\activate.bat
pip install -U ctranslate2 faster-whisper
```
Либо запустите с `--engine transformers` (если ставили `--with-transformers`).

**`Could not locate cudnn_ops64_9.dll` / `cublas64_12.dll`**
```bat
pip install --force-reinstall nvidia-cublas-cu12 nvidia-cudnn-cu12
```
`transcribe.py` сам подхватывает DLL из этих пакетов.

**`Torch not compiled with CUDA enabled`**
Поставлен CPU-torch. Переустановите:
```bat
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Русский текст в консоли «кракозябрами»**
Скрипты делают `chcp 65001`, а Python переключает stdout в UTF-8. Сам файл
`transcript.txt` всегда в UTF-8 — открывайте в VS Code / Notepad++.

**Антивирус/Defender ругается на скачивание модели**
Это Hugging Face Hub; добавьте папку проекта в исключения при необходимости.

**Модель долго скачивается в первый раз**
Нормально: `large-v3-turbo` ~1.6 ГБ, кэш в `%USERPROFILE%\.cache\huggingface`.
