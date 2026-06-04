# SCYTH-J: Smart Classification of Your Texts in Japanese


# Проект на Python
![Рекомендуемая версия: Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SCYTH-J** — веб-приложение на основе нейронных сетей для классификации японских текстов по динамически задаваемым пользовательским темам. Проект использует полученную с помощью глубокого обучения с учителем квантованную модель и методы обработки естественного языка для анализа и рубрицирования текстов.

---

## 📌 Описание
SCYTH-J позволяет:
- Анализировать японский текст и определять его тематику.
- Сравнивать тексты с эталонными векторами категорий.
- Добавлять/удалять эталонные образцы.

  ![Интерфейс SCYTH-J](app.gif)

---

## 🚀 Установка

### **Требования**
- Python 3.11.9
- Виртуальное окружение (`venv`)
- 3.2+ ГБ свободного места (для модели)

### **Инструкция по установке**

1. **Склонируйте репозиторий**:
   ```bash
   git clone https://github.com/jhgudleik/scyth_j.git
   cd scyth-j
   ```
2. **Используйте .bat файл**:
Дважды щелкните по init.bat. При первом запуске скачается модель, будут установленны зависимости и приложение откроется в браузере по адресу `http://127.0.0.1:port`.
При всех последующих запусках bat-файла приложение так же будет открываться в окне браузера.

   **Или самостоятельно...**:

3. **Создайте и активируйте виртуальное окружение**:
   ```bash
   python -m venv scyth_j_venv
   ```
   - **Windows**:
     ```batch
     scyth_j_venv\Scripts\activate
     ```
   - **Linux/Mac**:
     ```bash
     source scyth_j_venv/bin/activate
     ```

4. **Установите зависимости** (автоматически загрузит `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```

5. **Скачайте модель**.
По адресу:   https://huggingface.co/Cyrilos/scyth_5_cpu_int8/

6. **Запустите приложение**:
   ```bash
   python app.py
   ```
   Приложение откроется в браузере по адресу `http://127.0.0.1:port`.


---

## 📂 Структура проекта

```

scyth-j/
│
├── app.gif                  # Демо
├── init.bat                 # Установка зависимостей, скачивание модели и запуск приложения
├── scyth.png                # Логотип приложения
├── standard/                # Каталог эталонных данных (заполняется пользователем)
├── standard_test/           # Каталог демонстрационных образцов (дублирует структуру эталонного каталога
│   ├── vectors.txt          # Список векторов категорий
│   ├── topics.txt           # Карта ID → тема
│   ├── 1.vec, 2.vec, ...    # Векторные представления текстов
│   └── 1.txt, 2.txt, ...    # Примеры текстов
│
├── requirements.txt         # Зависимости (pip)
├── scyth_5_cpu_int8.pth     # Квантованная модель (автоматически скачивается)
├── app.py                   # Основной скрипт приложения
├── download_model.py        # Скачивание модели
├── README.md                # Этот файл
└── LICENSE                  # Лицензия проекта
```

---

## 🛠 Использование

### **Вкладка "Рубрикатор"**
1. Введите японский текст в поле `Новый текст` или выберите файл.
2. Нажмите кнопку **Рубрицировать**.
3. Приложение выведет список найденных категорий с вероятностью совпадения.
   
### **Добавление нового образца**
1. Нажмите **Добавить образец**.
2. Выберите категорию или введите пользовательскую.
3. Нажмите **Сохранить**.
4. Образец добавляется в базу и становится доступен для последующего анализа.
   
### **Управление категориями**
- Вкладка **Образцы** позволяет:
  - Сортировать по имени или теме.
  - Удалять ненужные образцы.

---

## 🔧 Настройки

Параметры модели и поиска задаются в начале файла `app.py`. Ключевые параметры:
| Параметр               | Описание                                                                 |
|------------------------|---------------------------------------------------------------------------|
| `TOP_K`                | Количество возвращаемых категорий (по умолчанию — `3`).               |
| `COSINE_THRESHOLD`     | Порог косинусного сходства (0.975 — рекомендуется для точности).     |
| `USE_FBGEMM`           | Использовать оптимизированный бекэнд для ускорения вычислений.         |

---

## 🔄 Скачивание модели
Модель `scyth_5_cpu_int8.pth` скачивается автоматически при первом запуске init.bat из репозитория Hugging Face:
```bash
repo_id="Cyrilos/scyth_5_cpu_int8"
```
Если скачивание не работает, убедитесь, что:
- Подключение к интернету стабильное.
- У вас есть доступ к Hugging Face Hub.

---

## 🛡 Лицензия

Проект распространяется под лицензией **MIT**. См. файл `LICENSE` для деталей.

---
## 📢 Контакты
- **Автор**: [Cyrilos](https://huggingface.co/Cyrilos)
- **Почта**: 📧 [jhgudleik@gmail.com]
- **GitHub**: [scyth-j-repo-link](https://github.com/jhgudleik/scyth_j)
- **Вопросы**: Открывайте `Issues` в репозитории.
---

# **English Version of Project Description**
![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SCYTH-J** is a **neural network-based web application** for classifying Japanese texts into user-defined dynamic topics. The system uses a **quantized teacher-trained deep learning model** and **natural language processing (NLP) techniques** to analyze and categorize texts.

---

## **📌 Overview**
SCYTH-J enables:
- Analysis of Japanese text to determine its thematic content.
- Comparison of texts against reference category vectors.
- Adding/removing reference samples.

---

## **🚀 Installation**

### **Requirements**
- Python **3.11.9**
- Virtual environment (`venv`)
- **3.2+ GB** free space (for the model)

### **Installation Steps**

#### **Option 1: Quick Setup (Using `.bat` file)**
1. Double-click **`init.bat`**.
   - On first run, it downloads the model, installs dependencies, and opens the app at `http://127.0.0.1:port`.
   - Subsequent runs will auto-launch the browser.

#### **Option 2: Manual Setup**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/jhgudleik/scyth_j.git
   cd scyth-j
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv scyth_j_venv
   ```
   - **Windows**:
     ```batch
     scyth_j_venv\Scripts\activate
     ```
   - **Linux/Mac**:
     ```bash
     source scyth_j_venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the model**:
   - From [Hugging Face](https://huggingface.co/Cyrilos/scyth_5_cpu_int8/).

5. **Run the application**:
   ```bash
   python app.py
   ```
   The app will open in your browser at `http://127.0.0.1:port`.

---

## **📂 Project Structure**
```
scyth-j/
│
├── app.gif                  # Demo screenshot
├── init.bat                 # Automates setup, model download, and app launch
├── scyth.png                # Project logo
├── standard/                # User-defined reference data (populated by users)
├── standard_test/           # Demo sample directory (mirrors structure of `standard/`)
│   ├── vectors.txt          # List of category vectors
│   ├── topics.txt           # ID → Topic mapping
│   ├── 1.vec, 2.vec, ...    # Vector files per category
│   └── 1.txt, 2.txt, ...    # Sample text files
│
├── requirements.txt         # Python dependencies
├── scyth_5_cpu_int8.pth     # Quantized model (auto-downloaded)
├── app.py                   # Main application script
├── download_model.py        # Script to download the model
├── README.md                # This file
└── LICENSE                  # MIT License
```

---

## **🛠 Usage**

### **"Categorizer" Tab**
1. Enter Japanese text in the **"New Text"** field or upload a file.
2. Click **"Categorize"**.
3. The app returns a ranked list of categories with match probabilities.

### **Adding a New Sample**
1. Click **"Add Sample"**.
2. Select or create a category.
3. Click **"Save"**.
4. The sample is added to the database for future analysis.

### **Managing Categories**
- The **"Samples"** tab allows:
  - Sorting by name or topic.
  - Deleting unwanted samples.

---

## **🔧 Configuration**
Key parameters are set in `app.py`:
| Parameter          | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `TOP_K`            | Number of returned categories (default: `3`).                              |
| `COSINE_THRESHOLD` | Cosine similarity threshold (recommended: `0.975` for accuracy).            |
| `USE_FBGEMM`       | Enable optimized backend for faster computations.                          |

---

## **🔄 Downloading the Model**
The model (`scyth_5_cpu_int8.pth`) is **automatically downloaded** via `init.bat` from Hugging Face:
```python
repo_id = "Cyrilos/scyth_5_cpu_int8"
```
If download fails:
- Ensure a **stable internet connection**.
- Verify **Hugging Face Hub access**.

---

## **🛡 License**
This project is released under the **MIT License**. See `LICENSE` for details.

---
## **📢 Contacts**
- **Author**: [Cyrilos](https://huggingface.co/Cyrilos)
- **Email**: 📧 [jhgudleik@gmail.com]
- **GitHub**: [scyth-j-repo](https://github.com/jhgudleik/scyth_j)
- **Issues**: Open a ticket in the repository.

---

### **🏷 Tags**
- [NLP](https://github.com/topics/nlp)
- [Japanese](https://github.com/topics/japanese)
- [Multilabel Classification](https://github.com/topics/multilabel-classification)
- [PyTorch](https://github.com/topics/pytorch)
- [Quantization](https://github.com/topics/quantization)
- [CPU](https://github.com/topics/cpu)
- [Semantic Search](https://github.com/topics/semantic-search)
- [Wikipedia](https://github.com/topics/wikipedia)
- [Embedding](https://github.com/topics/embedding)

---
© 2024 SCYTH-J. All rights reserved