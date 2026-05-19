"""
src/predict.py — Predicción de entidades sobre texto de facturas.
"""

import spacy
import re
import os


def load_model(model_path: str = "models/modelo_facturas"):
    """Carga el modelo NER entrenado desde disco."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{model_path}'. "
            "Ejecuta primero: python -m src.train_ner"
        )
    return spacy.load(model_path)


def predict_entities(text: str, nlp=None, model_path: str = "models/modelo_facturas"):
    """
    Detecta entidades PRODUCTO, CANTIDAD y PRECIO en el texto.

    Args:
        text: Texto de la factura (limpio)
        nlp: Modelo spaCy ya cargado (opcional, para reusar en Streamlit)
        model_path: Ruta al modelo si no se pasa nlp

    Returns:
        Lista de dicts: [{"text": ..., "label": ..., "start": ..., "end": ...}]
    """
    if nlp is None:
        nlp = load_model(model_path)

    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })

    return entities


def apply_hybrid_rules(text: str, entities: list) -> list:
    """
    Aplica reglas híbridas para mejorar la extracción cuando el NER falla.
    Detecta patrones que el modelo puede haber omitido.

    Reglas:
    - Tokens con "$" o que son solo dígitos largos (>4) → PRECIO candidato
    - Tokens que son dígitos cortos (1-3) → CANTIDAD candidata
    """
    # Etiquetas ya detectadas
    detected_labels = {ent["label"] for ent in entities}

    # Regla 1: Si no hay PRECIO, buscar número largo
    if "PRECIO" not in detected_labels:
        matches = re.finditer(r"\$?\s*(\d{4,})", text)
        for m in matches:
            entities.append({
                "text": m.group(),
                "label": "PRECIO",
                "start": m.start(),
                "end": m.end(),
            })

    # Regla 2: Si no hay CANTIDAD, buscar número corto seguido de UND/UDS/x
    if "CANTIDAD" not in detected_labels:
        matches = re.finditer(r"\b(\d{1,3})\s*(UND|UDS|X|UNID|PZAS?)\b", text, re.IGNORECASE)
        for m in matches:
            entities.append({
                "text": m.group(1),
                "label": "CANTIDAD",
                "start": m.start(1),
                "end": m.end(1),
            })

    return entities
