@echo off
chcp 65001 >nul 2>&1
:: Работаем с текущей папкой (где лежит скрипт)

:: Проверяем Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Install it from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Создаем venv (если его нет)
if not exist "scyth_j_venv" (
    echo Creating virtual environment...
    python -m venv scyth_j_venv
    if %errorlevel% neq 0 (
        echo Error when creating venv.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

:: Активируем venv (используем activate.bat, а не просто activate)
call scyth_j_venv\Scripts\activate.bat || (
    echo Error: Failed to activate venv.
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

    echo Found requirements.txt file.
    echo.

    if "%LOG_INSTALL%"=="1" (
        echo Установка зависимостей ^(лог: install_log.txt^)...
        pip install -r "%REQ_FILE%" --upgrade > install_log.txt 2>&1
    ) else (
        echo Installing dependencies...
        pip install -r "%REQ_FILE%" --upgrade
    )

    if %errorlevel% neq 0 (
        echo.
        echo Error occurred during dependency installation.
        if "%LOG_INSTALL%"=="1" (
            echo Проверьте файл install_log.txt
        )
        pause
        exit /b 1
    )

) else (
    echo Error: requirements.txt file not found in current directory.
    echo Check that the script is running from the correct directory.
    pause
    exit /b 1
)
:: Настраиваем .gitignore
if exist ".gitignore" (
    findstr /i /c:"scyth_j_venv" ".gitignore" >nul
    if %errorlevel% equ 0 (
        echo scyth_j_venv already in .gitignore.
    ) else (
        echo Adding scyth_j_venv to .gitignore...
        echo scyth_j_venv >> .gitignore
    )
) else (
    echo Creating .gitignore and adding scyth_j_venv...
    echo scyth_j_venv > .gitignore
)

echo Done!
echo Activated environment: scyth_j_venv
echo Current packages:
pip list


:: =========================================================
:: СКАЧИВАНИЕ МОДЕЛИ
:: =========================================================

set "MODEL_DOWNLOADER=download_model.py"

if exist "%MODEL_DOWNLOADER%" (
    echo.
    echo Checking model availability...
    python "%MODEL_DOWNLOADER%"

    if %errorlevel% neq 0 (
        echo Error downloading the model.
        pause
        exit /b 1
    )
) else (
    echo Error: file %MODEL_DOWNLOADER% not found.
    pause
    exit /b 1
)

:: =========================================================
:: ЗАПУСК ПРИЛОЖЕНИЯ
:: =========================================================

set "APP_FILE=app.py"

if exist "%APP_FILE%" (
    echo.
    echo Running application app.py...
    python "%APP_FILE%"

    if %errorlevel% neq 0 (
        echo Error running application app.py
        pause
        exit /b 1
    )
) else (
    echo Warning: app.py file not found.
)

echo.
echo Press any key to exit...
pause >nul