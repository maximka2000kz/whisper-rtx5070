#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper-транскрипция, оптимизированная под RTX 5070 (Blackwell, 12 ГБ VRAM).
Кроссплатформенно: Windows и Linux.

Основной движок — faster-whisper (CTranslate2): самый быстрый на GPU.
  * модель large-v3-turbo, батчевый инференс, VAD
  * mp4/mkv/mp3/m4a/wav напрямую (через PyAV, без внешнего ffmpeg)
  * ссылки Rutube/YouTube через yt-dlp

Резервный движок — Transformers (AutoProcessor + AutoModelForSpeechSeq2Seq).

Примеры:
    python transcribe.py video.mp4 -o transcript.txt
    python transcribe.py video.mp4 -o out.txt --srt out.srt
    python transcribe.py "https://rutube.ru/video/<id>/" -o out.txt
    python transcribe.py video.mp4 --engine transformers
    python transcribe.py video.mp4 --compute-type int8_float16 --batch 24
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time

DEFAULT_MODEL = "large-v3-turbo"          # имя модели faster-whisper
DEFAULT_HF_MODEL = "openai/whisper-large-v3-turbo"
SR = 16000

_DLL_HANDLES = []  # держим ссылки, чтобы каталоги DLL не закрылись


def log(msg):
    print(msg, flush=True)


def setup_console_utf8():
    """Чтобы русский текст не падал в Windows-консоли (cp1251)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def setup_windows_dlls():
    """Windows: подключаем DLL cuBLAS/cuDNN из пакетов nvidia-*-cu12."""
    if os.name != "nt":
        return
    try:
        import site
        bases = []
        try:
            bases += site.getsitepackages()
        except Exception:  # noqa: BLE001
            pass
        if site.getusersitepackages():
            bases.append(site.getusersitepackages())
        for base in bases:
            for sub in ("cublas", "cudnn", "cuda_runtime"):
                for p in glob.glob(os.path.join(base, "nvidia", sub, "bin")):
                    try:
                        _DLL_HANDLES.append(os.add_dll_directory(p))
                        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                    except Exception:  # noqa: BLE001
                        pass
    except Exception:  # noqa: BLE001
        pass


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def run_ytdlp(url: str, outdir: str) -> str:
    """Скачивает лучший аудиопоток (без конвертации -> ffmpeg не нужен)."""
    log(f"[i] Скачиваю аудио: {url}")
    tpl = os.path.join(outdir, "source.%(ext)s")
    cmd = [sys.executable, "-m", "yt_dlp",
           "-f", "bestaudio/best", "-o", tpl, "--no-playlist", url]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("[x] Не найден yt-dlp. Установите: pip install yt-dlp")
    except subprocess.CalledProcessError:
        sys.exit("[x] yt-dlp не смог скачать. Проверьте ссылку/доступ к видео.")
    files = [f for f in glob.glob(os.path.join(outdir, "source.*"))
             if not f.endswith(".part")]
    if not files:
        sys.exit("[x] yt-dlp не создал аудиофайл")
    return files[0]


def ffmpeg_extract(path: str, outdir: str) -> str:
    """Достаёт звук в 16кГц моно WAV (нужно только резервному движку)."""
    if not shutil.which("ffmpeg"):
        return path
    out = os.path.join(outdir, "extracted.wav")
    cmd = ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
           "-f", "wav", "-loglevel", "error", out]
    try:
        subprocess.run(cmd, check=True)
        return out
    except subprocess.CalledProcessError:
        return path


def write_txt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            t = seg["text"].strip()
            if t:
                f.write(t + "\n")


def fmt_ts(sec):
    ms = int(round(float(sec) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, path):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            t = seg["text"].strip()
            if not t:
                continue
            n += 1
            f.write(f"{n}\n{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}\n{t}\n\n")
    return n


# --------------------------------------------------------------------------
# Движок 1: faster-whisper (быстрый, основной)
# --------------------------------------------------------------------------
def run_faster(args, audio_path):
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    log(f"[i] faster-whisper | model={args.model} | compute={args.compute_type} | batch={args.batch}")

    candidates = [args.compute_type]
    for ct in ("float16", "int8_float16", "int8"):
        if ct not in candidates:
            candidates.append(ct)

    model = None
    last_err = None
    for ct in candidates:
        try:
            model = WhisperModel(args.model, device="cuda", compute_type=ct,
                                 cpu_threads=0, num_workers=1)
            if ct != args.compute_type:
                log(f"[!] compute_type={args.compute_type} не подошёл, использую {ct}")
            args.compute_type = ct
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            log(f"[!] compute_type={ct} не сработал: {e}")

    if model is None:
        raise RuntimeError(f"faster-whisper: не удалось инициализировать CUDA ({last_err})")

    pipe = BatchedInferencePipeline(model=model)
    segments_iter, info = pipe.transcribe(
        audio_path,
        batch_size=args.batch,
        language=args.lang,
        task=args.task,
        beam_size=args.beam,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )

    dur = info.duration or 0
    log(f"[i] Язык: {info.language} (p={info.language_probability:.2f}) | длительность: {dur/60:.1f} мин")

    pbar = None
    try:
        from tqdm import tqdm
        pbar = tqdm(total=dur, unit="s", desc="STT", dynamic_ncols=True)
    except Exception:  # noqa: BLE001
        pass

    segments = []
    last = 0.0
    for s in segments_iter:
        segments.append({"start": s.start, "end": s.end, "text": s.text})
        if pbar is not None:
            pbar.update(max(0.0, s.end - last))
            last = s.end
    if pbar is not None:
        pbar.close()
    return segments


# --------------------------------------------------------------------------
# Движок 2: Transformers (резервный)
# --------------------------------------------------------------------------
def run_transformers(args, audio_path):
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    model_id = args.hf_model

    log(f"[i] transformers | model={model_id} | device={'cuda' if use_cuda else 'cpu'} | dtype={dtype}")

    processor = AutoProcessor.from_pretrained(model_id)
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="auto")
    except TypeError:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True, device_map="auto")
    model.eval()

    pipe = pipeline("automatic-speech-recognition", model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor)

    res = pipe(audio_path, chunk_length_s=30, stride_length_s=5,
               batch_size=args.batch, return_timestamps=True,
               generate_kwargs={"language": args.lang, "task": args.task})

    segments = []
    for ch in res.get("chunks", []) or []:
        ts = ch.get("timestamp") or (0, 0)
        segments.append({"start": ts[0] or 0, "end": ts[1] or ts[0] or 0,
                         "text": (ch.get("text") or "")})
    if not segments and res.get("text"):
        segments = [{"start": 0, "end": 0, "text": res["text"]}]
    return segments


def main():
    setup_console_utf8()
    setup_windows_dlls()

    p = argparse.ArgumentParser(
        description="Whisper-транскрипция для RTX 5070 (faster-whisper / transformers)")
    p.add_argument("input", help="Файл (mp4/mkv/mp3/m4a/wav...) или ссылка (Rutube/YouTube)")
    p.add_argument("-o", "--output", default="transcript.txt")
    p.add_argument("--srt", default=None, help="Сохранить субтитры .srt")
    p.add_argument("--engine", default="auto", choices=["auto", "faster", "transformers"],
                   help="auto: faster-whisper, при сбое transformers")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Модель faster-whisper")
    p.add_argument("--hf-model", default=DEFAULT_HF_MODEL, help="Модель HF для transformers")
    p.add_argument("--lang", default="ru")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--batch", type=int, default=16, help="Батч (12 ГБ: 16-24)")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--compute-type", default="float16",
                   help="float16 | int8_float16 | int8")
    p.add_argument("--keep-temp", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    tmpdir = tempfile.mkdtemp(prefix="whisper_")
    try:
        if is_url(args.input):
            audio_path = run_ytdlp(args.input, tmpdir)
        else:
            if not os.path.isfile(args.input):
                sys.exit(f"[x] Файл не найден: {args.input}")
            audio_path = args.input

        segments = None
        if args.engine in ("auto", "faster"):
            try:
                segments = run_faster(args, audio_path)
            except Exception as e:  # noqa: BLE001
                if args.engine == "faster":
                    raise
                log(f"[!] faster-whisper не сработал: {e}")
                log("[i] Переключаюсь на transformers (резервный движок)")

        if segments is None:
            if args.engine == "faster":
                sys.exit("[x] faster-whisper недоступен")
            src = ffmpeg_extract(audio_path, tmpdir)
            segments = run_transformers(args, src)

        write_txt(segments, args.output)
        chars = sum(len(s["text"]) for s in segments)
        log(f"[+] Текст: {args.output} ({chars} символов, {len(segments)} сегментов)")
        if args.srt:
            n = write_srt(segments, args.srt)
            log(f"[+] Субтитры: {args.srt} ({n} реплик)")
        log(f"[+] Готово за {time.time() - t0:.1f} сек")
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            log(f"[i] Временные файлы: {tmpdir}")


if __name__ == "__main__":
    main()
