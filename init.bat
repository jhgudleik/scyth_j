@echo off
chcp 65001 >nul 2>&1
:: Работаем с текущей папкой (где лежит скрипт)

:: Проверяем Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ошибка: Python не найден. Установите его с https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Создаем venv (если его нет)
if not exist "scyth_j_venv" (
    echo Создание виртуального окружения...
    python -m venv scyth_j_venv
    if %errorlevel% neq 0 (
        echo Ошибка при создании venv.
        pause
        exit /b 1
    )
) else (
    echo Виртуальное окружение уже существует.
)

:: Активируем venv (используем activate.bat, а не просто activate)
call scyth_j_venv\Scripts\activate.bat || (
    echo Ошибка: Не удалось активировать venv.
    pause
    exit /b 1
)

:: =========================================================
:: УСТАНОВКА ЗАВИСИМОСТЕЙ
:: =========================================================

:: 0 = показывать установку в консоли
:: 1 = писать только в install_log.txt
set "LOG_INSTALL=0"

set "REQ_FILE=requirements.txt"

if exist "%REQ_FILE%" (

    echo Найден файл requirements.txt.
    echo.

    if "%LOG_INSTALL%"=="1" (
        echo Установка зависимостей ^(лог: install_log.txt^)...
        pip install -r "%REQ_FILE%" --upgrade > install_log.txt 2>&1
    ) else (
        echo Установка зависимостей...
        pip install -r "%REQ_FILE%" --upgrade
    )

    if %errorlevel% neq 0 (
        echo.
        echo Ошибка при установке зависимостей.
        if "%LOG_INSTALL%"=="1" (
            echo Проверьте файл install_log.txt
        )
        pause
        exit /b 1
    )

) else (
    echo Ошибка: Файл requirements.txt не найден в текущей папке.
    echo Проверьте, что скрипт запускается из корректной директории.
    pause
    exit /b 1
)
:: Настраиваем .gitignore
if exist ".gitignore" (
    findstr /i /c:"scyth_j_venv" ".gitignore" >nul
    if %errorlevel% equ 0 (
        echo scyth_j_venv уже в .gitignore.
    ) else (
        echo Добавляем scyth_j_venv в .gitignore...
        echo scyth_j_venv >> .gitignore
    )
) else (
    echo Создаем .gitignore и добавляем scyth_j_venv...
    echo scyth_j_venv > .gitignore
)

echo Готово!
echo Активированное окружение: scyth_j_venv
echo Текущие пакеты:
pip list

echo Готово!
echo Активированное окружение: scyth_j_venv
echo Текущие пакеты:
pip list

:: =========================================================
:: СКАЧИВАНИЕ МОДЕЛИ
:: =========================================================

set "MODEL_DOWNLOADER=download_model.py"

if exist "%MODEL_DOWNLOADER%" (
    echo.
    echo Проверка наличия модели...
    python "%MODEL_DOWNLOADER%"

    if %errorlevel% neq 0 (
        echo Ошибка при скачивании модели.
        pause
        exit /b 1
    )
) else (
    echo Ошибка: файл %MODEL_DOWNLOADER% не найден.
    pause
    exit /b 1
)

:: =========================================================
:: ЗАПУСК ПРИЛОЖЕНИЯ
:: =========================================================

set "APP_FILE=app.py"

if exist "%APP_FILE%" (
    echo.
    echo Запуск приложения app.py...
    python "%APP_FILE%"

    if %errorlevel% neq 0 (
        echo Ошибка при запуске приложения app.py
        pause
        exit /b 1
    )
) else (
    echo Предупреждение: Файл app.py не найден.
)

echo.
echo Нажмите любую клавишу для выхода...
pause >nul