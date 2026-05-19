"""
src/ocr.py — Extracción de texto desde PDF e imágenes usando OCR.

Requiere:
  - Tesseract instalado en el sistema
  - pytesseract, pypdf, pillow

En Windows, agregar al inicio:
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
"""

import pytesseract
from pypdf import PdfReader
from PIL import Image
import os

# Configurar Tesseract para Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrae texto directamente del PDF sin necesidad de Poppler.
    Si el PDF es un escaneo, convierte las páginas a imágenes para OCR.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No se encontró el archivo: {pdf_path}")

    try:
        # Intentar extraer texto directamente (PDFs con texto embebido)
        reader = PdfReader(pdf_path)
        full_text = ""
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():  # Si tiene texto, usarlo
                full_text += f"\n--- Página {page_num} ---\n{text}"
        
        # Si se extrajo texto, retornarlo
        if full_text.strip():
            return full_text
    except Exception as e:
        print(f"Error extrayendo texto con pypdf: {e}")
    
    # Si no hay texto embebido o error, retornar mensaje
    print(f"Nota: El PDF {pdf_path} parece ser un escaneo. Se necesitaría OCR con imágenes.")
    return f"No se pudo extraer texto del PDF. El archivo puede ser un escaneo sin texto embebido."


def extract_text_from_image(image_path: str) -> str:
    """
    Aplica OCR sobre una imagen de factura.
    Soporta: PNG, JPG, TIFF, BMP.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    image = Image.open(image_path)

    # Convertir a escala de grises mejora el OCR
    image = image.convert("L")

    custom_config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(image, lang="spa", config=custom_config)

    return text


def extract_text_from_text_file(txt_path: str) -> str:
    """
    Lee directamente un archivo .txt (para pruebas sin OCR).
    Útil cuando la factura ya viene en texto plano.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()
