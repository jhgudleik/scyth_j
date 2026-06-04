from huggingface_hub import hf_hub_download
import os

# Путь к текущей папке (где лежит скрипт)
save_dir = "."
filename = "scyth_5_cpu_int8.pth"
full_save_path = os.path.join(save_dir, filename)  # Полный путь к файлу

# Проверяем, существует ли файл уже в папке
if os.path.exists(full_save_path):
    print(f"✅ Файл {filename} уже существует в папке: {full_save_path}")
    print(f"Размер файла: {os.path.getsize(full_save_path) / (1024 * 1024):.2f} MB")
else:
    print(f"🔍 Файл {filename} не найден в папке. Скачивание может занять до 7 мин. Начинаем скачивание...")
    try:
        # Скачиваем файл
        downloaded_file = hf_hub_download(
            repo_id="Cyrilos/scyth_5_cpu_int8",
            filename=filename,
            local_dir=save_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✅ Файл успешно скачан: {downloaded_file}")
        print(f"Размер скачанного файла: {os.path.getsize(downloaded_file) / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"❌ Ошибка при скачивании: {e}")
        if os.path.exists(full_save_path):
            print(f"⚠️ Однако файл {filename} уже существует в папке (возможно, частично скачан).")
        else:
            print(f"⚠️ Файл {filename} не был скачан.")
