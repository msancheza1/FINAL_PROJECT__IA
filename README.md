# FINAL_PROJECT__IA
AGENT IA 


# 📄 Invoice AI — Extracción Automática de Facturas

Sistema de IA para extraer automáticamente productos, cantidades y precios
desde facturas de proveedores, generando un archivo CSV/JSON listo para
importar al sistema POS de inventario.

**Proyecto:** Machine Learning — Universidad  
**Integrantes:** Mariana Sanchez, Helen Yanes, Andres Echeverri

---

## 🏗️ Arquitectura

```
PDF/Imagen → OCR (Tesseract) → Limpieza de texto → NER (spaCy) → CSV/JSON → Streamlit UI
```

El modelo aprende a reconocer tres entidades en el texto de facturas:
- **PRODUCTO** — Nombre del artículo
- **CANTIDAD** — Número de unidades
- **PRECIO** — Valor unitario

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/msancheza1/FINAL_PROJECT__IA.git
cd invoice_ai
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias Python

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

### 4. Instalar Tesseract OCR

- **Windows:** Descargar desde https://github.com/tesseract-ocr/tesseract  
  Luego descomentar en `src/ocr.py`:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

- **Ubuntu/Debian:**
  ```bash
  sudo apt install tesseract-ocr tesseract-ocr-spa
  ```

- **Mac:**
  ```bash
  brew install tesseract tesseract-lang
  ```

---

## 🚀 Uso

### Paso 1: Agregar datos de entrenamiento

Edita `data/annotations/train_data.py` con tus facturas reales, o genera datos sintéticos:

```bash
cd data/annotations
python generate_train_data.py
```

### Paso 2: Entrenar el modelo

```bash
python -m src.train_ner
```

El modelo se guarda en `models/modelo_facturas/`.

### Paso 3: Ejecutar la interfaz

```bash
streamlit run app/streamlit_app.py
```

Abre tu navegador en http://localhost:8501

---

## 📁 Estructura del proyecto

```
invoice_ai/
│
├── app/
│   └── streamlit_app.py          # Interfaz Streamlit (producto final)
│
├── data/
│   ├── raw/                      # Facturas originales (PDF, imágenes)
│   ├── processed/                # Texto extraído por OCR
│   └── annotations/
│       ├── train_data.py         # Datos de entrenamiento NER
│       └── generate_train_data.py
│
├── models/
│   └── modelo_facturas/          # Modelo entrenado spaCy
│
├── notebooks/                    # Experimentación y análisis ML
│   ├── 01_EDA.ipynb              # Análisis exploratorio de datos
│   ├── 02_OCR_Experiments.ipynb  # Pruebas de extracción OCR
│   ├── 03_NER_Training.ipynb     # Entrenamiento + curva de loss
│   └── 04_Evaluation.ipynb       # Precision, Recall, F1 por entidad
│
├── results/
│   ├── figures/                  # Gráficas generadas por notebooks
│   ├── metrics/                  # ner_scores.json
│   └── outputs/                  # CSV/JSON de ejemplo
│
├── sample_invoices/              # Facturas de prueba (anonimizadas)
│
├── src/
│   ├── ocr.py                    # Extracción con Tesseract
│   ├── preprocess.py             # Limpieza y normalización
│   ├── train_ner.py              # Script de entrenamiento
│   ├── predict.py                # Inferencia + reglas híbridas
│   └── export.py                 # Exportación a CSV/JSON
│
├── .gitignore
├── requirements.txt
└── README.md
```

### ¿Por qué esta estructura?

| Carpeta | Propósito |
|---|---|
| `notebooks/` | Proceso científico / ML (experimentación reproducible) |
| `src/` | Lógica modular reutilizable |
| `app/` | Producto final funcional |
| `models/` | Modelo entrenado (checkpoints) |
| `results/` | Evaluación y gráficas |
| `sample_invoices/` | Casos de prueba reales |

---

## 📊 Evaluación

El modelo se evalúa con **F1-score por entidad** (estándar para NER):

| Entidad  | Precision | Recall | F1     |
|----------|-----------|--------|--------|
| PRODUCTO | ~0.85     | ~0.83  | ~0.84  |
| CANTIDAD | ~0.90     | ~0.88  | ~0.89  |
| PRECIO   | ~0.92     | ~0.90  | ~0.91  |

*Resultados aproximados con datos sintéticos. Mejoran con facturas reales.*

---

## 📚 Referencias

1. Lample et al. (2016). *Neural Architectures for Named Entity Recognition.* https://arxiv.org/abs/1603.01360
2. spaCy — Training NER models. https://spacy.io/usage/training
3. Tesseract OCR. https://github.com/tesseract-ocr/tesseract
