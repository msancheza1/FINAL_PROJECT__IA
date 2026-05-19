"""
src/preprocess.py — Limpieza y normalización del texto de facturas.
"""

import re


def clean_text(text: str) -> str:
    """
    Limpieza básica del texto extraído por OCR:
    - Elimina saltos de línea excesivos
    - Colapsa espacios múltiples
    - Normaliza caracteres comunes mal leídos por OCR
    """
    # Reemplazar saltos de línea por espacio
    text = text.replace("\n", " ").replace("\r", " ")

    # Normalizar confusiones comunes de OCR en facturas
    ocr_fixes = {
        "0": ["O", "o"],  # cero confundido con letra O
        "$": ["S$", "5$"],
    }

    # Colapsar espacios múltiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalización más agresiva para mejorar NER:
    - Convierte a mayúsculas
    - Elimina caracteres especiales excepto $, . y ,
    """
    text = text.upper()
    # Conservar solo caracteres alfanuméricos, espacios y símbolos relevantes
    text = re.sub(r"[^A-Z0-9\s\$\.\,\:\(\)\/\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_lines(text: str) -> list:
    """
    Divide el texto en líneas candidatas a ser ítems de factura.
    Una línea es un segmento que potencialmente contiene producto, cantidad y precio.
    """
    # Primero dividimos por patrones que indican nuevos ítems:
    # - Una cantidad al inicio (ej: "2 CASCO...")
    # - Un código de producto (ej: "ACP-001 CASCO...")
    lines = []
    raw_lines = text.split("  ")  # Doble espacio como separador de líneas OCR

    for line in raw_lines:
        line = line.strip()
        if len(line) > 5:  # Ignorar fragmentos muy cortos
            lines.append(line)

    return lines


def extract_candidate_lines(text: str) -> list:
    """
    Filtra líneas que probablemente son ítems de factura.
    Criterio: contiene al menos un número (precio o cantidad).
    """
    lines = split_into_lines(text)
    candidates = []

    for line in lines:
        # Verificar que la línea tiene al menos un número
        if re.search(r"\d+", line):
            candidates.append(line)

    return candidates
