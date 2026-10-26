# 🫘 Dry Bean Classification

**Порівняння PyTorch, TensorFlow, PaddlePaddle та MindSpore**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-0062D1)
![MindSpore](https://img.shields.io/badge/MindSpore-5B5FC7)

> **4 фреймворки · 1 архітектура · 20 запусків · 7 класів**

Проєкт досліджує, як чотири фреймворки глибокого навчання розв'язують **однакову задачу класифікації сортів квасолі** на наборі даних Dry Bean. Для кожного фреймворку використовується однакова нейронна мережа, а результати п'яти запусків порівнюються за якістю класифікації та часом навчання.

## 🚀 Запуск проєкту

Відкрийте **PowerShell у кореневій папці проєкту**. Для відтворення експерименту використовуйте **Python 3.12**.

### 1. Створіть віртуальне середовище

```powershell
python -m venv .venv
```

### 2. Встановіть залежності

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Запустіть усі експерименти

```powershell
.\.venv\Scripts\python.exe run_all.py
```

Скрипт `run_all.py` підготує дані, запустить **20 експериментів** (5 для кожного фреймворку), збереже метрики та створить графіки. Під час повторного запуску вже перевірені результати пропускаються.

> **Важливо:** файл `Dry_Bean_Dataset.xlsx` має знаходитися в корені проєкту. Активувати віртуальне середовище окремою командою не потрібно.

## 📊 Результати

Середні показники за п'ятьма запусками кожного фреймворку:

| Фреймворк | Точність (Accuracy) | Macro-F1 | Час навчання |
|:--|--:|--:|--:|
| **PyTorch** | 92,33 % | 93,52 % | 5,38 с |
| **TensorFlow** | 92,36 % | 93,54 % | 4,75 с |
| **PaddlePaddle** | 92,36 % | 93,54 % | 4,44 с |
| **MindSpore** | 92,34 % | 93,52 % | 21,05 с |

Усі фреймворки показали **близьку якість класифікації**. Час виконання залежить від умов конкретного експерименту; наведені цифри не визначають універсально найкращий інструмент.

## 🧬 Дані та нейронна мережа

Використовується [**Dry Bean Dataset**](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) з репозиторію UCI Machine Learning Repository.

| Параметр | Значення |
|:--|:--|
| Кількість зразків | 13 611 |
| Кількість ознак | 16 |
| Кількість класів | 7 |
| Поділ даних | 70 % / 15 % / 15 % |
| Архітектура | `16 → 16 → 32 → 64 → 7` |
| Навчання | 100 епох, Adam |

Для порівняння використовуються спільні початкові ваги, порядок навчальних прикладів та однакові підготовлені дані.

## 📁 Результати запуску

| Шлях | Що містить |
|:--|:--|
| `results_v2/all_runs.csv` | Метрики всіх 20 запусків |
| `results_v2/comparison_mean_sd.csv` | Середні значення та стандартні відхилення |
| `results_v2/comparison_formatted.csv` | Підсумкова таблиця порівняння |
| `figures/` | Графіки навчання та матриці помилок |
| `results_v2/logs/` | Журнали запусків |

## 🛠️ Додаткові команди

**Один запуск PyTorch:**

```powershell
.\.venv\Scripts\python.exe train_pytorch.py --seed 42
```

**П'ять запусків лише PyTorch:**

```powershell
.\.venv\Scripts\python.exe run_all.py --framework PyTorch
```

**Перевірка отриманих результатів:**

```powershell
.\.venv\Scripts\python.exe verify_results.py
```

**Повторне створення підсумкових таблиць і графіків:**

```powershell
.\.venv\Scripts\python.exe compare_results.py
```

## 🗂️ Структура проєкту

```text
DryBean/
├── common.py                 # Спільні налаштування та метрики
├── prepare_data.py           # Підготовка набору даних
├── train_pytorch.py          # Реалізація PyTorch
├── train_tensorflow.py       # Реалізація TensorFlow
├── train_paddlepaddle.py     # Реалізація PaddlePaddle
├── train_mindspore.py        # Реалізація MindSpore
├── run_all.py                # Запуск експериментів
├── compare_results.py        # Підсумкові таблиці й графіки
├── verify_results.py        # Перевірка результатів
├── requirements.txt          # Залежності
└── Dry_Bean_Dataset.xlsx     # Вхідні дані
```

---

**Джерело даних:** Dry Bean Dataset, UCI Machine Learning Repository (2020). [DOI: 10.24432/C50S4B](https://doi.org/10.24432/C50S4B).
