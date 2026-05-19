"""
src/export.py — Convierte entidades detectadas a DataFrame, CSV y JSON.
"""

import pandas as pd
import json
import re


def entities_to_dataframe(entities: list) -> pd.DataFrame:
    """
    Agrupa entidades por tipo y construye un DataFrame de inventario.

    Asume que las entidades llegan en orden secuencial
    (producto → cantidad → precio) para cada ítem.

    Args:
        entities: Lista de dicts con keys: text, label

    Returns:
        DataFrame con columnas: producto, cantidad, precio
    """
    items = []
    current = {"producto": None, "cantidad": None, "precio": None}

    # Palabras comunes que NO son productos (ruido)
    noise_words = {
        "TIENDAS", "TIENDA", "FECHA", "REGIMEN", "COMUN", "RADICAR",
        "FACTURA", "ELECTRONICA", "COMPRAS", "FERRETERIA", "ORC", "ORDENES",
        "PRINCIPAL", "PAGINA", "PAGE", "GINA", "DE", "CRA", "ENTREGA",
        "FACTURA", "ELECTRONICA", "FECHA", "FACTURADOR", "RESPONSABLE",
        "TELEFONO", "DIRECCION", "CIUDAD", "PAIS", "SUBTOTAL", "IVA",
        "TOTAL", "TOTAL FACTURA", "PAGO", "METODO", "DESCRIPCION",
        "CONCEPTO", "NIT", "CC", "CEDULA", "NOMBRE", "EMPRESA"
    }

    for ent in entities:
        label = ent["label"].upper()
        text = clean_entity_value(ent["text"], label)

        if label == "PRODUCTO":
            # Filtrar ruido
            if is_valid_product(text, noise_words):
                # Si ya teníamos un ítem en curso, guardarlo
                if current["producto"] is not None:
                    items.append(current.copy())
                    current = {"producto": None, "cantidad": None, "precio": None}
                current["producto"] = text

        elif label == "CANTIDAD":
            current["cantidad"] = text

        elif label == "PRECIO":
            current["precio"] = text

    # Guardar último ítem
    if current["producto"] is not None:
        items.append(current)

    if not items:
        return pd.DataFrame(columns=["producto", "cantidad", "precio"])

    df = pd.DataFrame(items)

    # Limpiar tipos de datos
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    df["precio"] = df["precio"].apply(parse_price)

    return df


def is_valid_product(text: str, noise_words: set) -> bool:
    """
    Valida si el texto es realmente un producto.
    Descarta ruido y datos irrelevantes.
    """
    if not text or len(text.strip()) < 3:
        return False
    
    # Si es solo números, no es producto
    if text.isdigit():
        return False
    
    # Si es una palabra de ruido conocida
    if text.upper() in noise_words:
        return False
    
    # Si tiene menos de 2 caracteres alfabéticos, probablemente sea ruido
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 2:
        return False
    
    return True


def clean_entity_value(text: str, label: str) -> str:
    """Limpieza específica por tipo de entidad."""
    text = text.strip()

    if label == "PRECIO":
        # Eliminar $, puntos de miles, etc.
        text = re.sub(r"[$\s]", "", text)
        text = text.replace(".", "").replace(",", "")

    elif label == "CANTIDAD":
        # Mantener solo el número
        match = re.search(r"\d+", text)
        if match:
            text = match.group()

    return text


def parse_price(value) -> float:
    """Convierte texto de precio a float."""
    if pd.isna(value) or value is None:
        return None
    try:
        cleaned = re.sub(r"[^\d]", "", str(value))
        return float(cleaned) if cleaned else None
    except Exception:
        return None


def dataframe_to_csv(df: pd.DataFrame, output_path: str = "inventario.csv") -> str:
    """Guarda el DataFrame como CSV listo para importar al POS."""
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def dataframe_to_json(df: pd.DataFrame, output_path: str = "inventario.json") -> str:
    """Guarda el DataFrame como JSON."""
    records = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return output_path
