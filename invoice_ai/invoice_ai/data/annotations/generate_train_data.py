"""
Generador de datos de entrenamiento para NER de facturas.
Ejecutar este script para producir train_data.py con ejemplos sintéticos
basados en patrones reales de facturas colombianas.
"""

import random
import re

# --- Vocabulario base (ampliar con datos reales de Accemotos) ---

PRODUCTOS = [
    "CASCO SHAFT XTR-902", "CASCO FOX V1", "CASCO LS2 FF320",
    "GUANTES FOX DIRTPAW", "GUANTES ANSWER", "GUANTES SHOT",
    "RODILLERA LEATT", "RODILLERA POD K4", "CODERAS ALPINESTARS",
    "IMPERMEABLE SHAFT", "CHAQUETA ALPINESTARS", "BOTA SIDI CROSSFIRE",
    "BOTA FOX COMP 5", "MALETERO GIVI V35", "MALETERO SHAD SH48",
    "PROTECTOR ESPALDA LEATT", "CHALECO PROTECTOR FOX",
    "CASCO HAWK HX-W312", "CASCO YOHE 953", "GUANTE MOTOWOLF",
    "SEGURO DE MERCANCIA", "FLETE", "DESCUENTO COMERCIAL",
]

CANTIDADES = [str(i) for i in range(1, 25)]

PRECIOS = [
    "12605", "45000", "58000", "62000", "75000", "80000",
    "95000", "110000", "125000", "145000", "180000", "218000",
    "229411", "320000", "450000",
]

# Plantillas de formato (variaciones reales en facturas)
PLANTILLAS = [
    "{producto} {cantidad} {precio}",
    "{cantidad} {producto} {precio}",
    "{producto} {cantidad} UND {precio}",
    "{producto} CANT: {cantidad} VR: {precio}",
    "DESCRIPCION: {producto} QTY {cantidad} PRECIO: {precio}",
    "{producto} x{cantidad} $ {precio}",
    "{cantidad} UND {producto} TOTAL ${precio}",
    "{producto} {cantidad} UND ${precio}",
    "{producto} ({cantidad}) VLR UNIT {precio}",
    "{cantidad} {producto} P.U. {precio}",
]

def find_span(text, substring):
    """Encuentra el span (inicio, fin) de una subcadena en el texto."""
    start = text.find(substring)
    if start == -1:
        return None
    return (start, start + len(substring))

def generate_example(plantilla, producto, cantidad, precio):
    """Genera un ejemplo de entrenamiento con sus anotaciones."""
    text = plantilla.format(
        producto=producto,
        cantidad=cantidad,
        precio=precio
    )

    entities = []

    span_p = find_span(text, producto)
    span_c = find_span(text, cantidad)
    span_pr = find_span(text, precio)

    if span_p:
        entities.append((span_p[0], span_p[1], "PRODUCTO"))
    if span_c:
        entities.append((span_c[0], span_c[1], "CANTIDAD"))
    if span_pr:
        entities.append((span_pr[0], span_pr[1], "PRECIO"))

    # Verificar que no haya solapamientos
    spans = sorted(entities, key=lambda x: x[0])
    valid = True
    for i in range(len(spans) - 1):
        if spans[i][1] > spans[i + 1][0]:
            valid = False
            break

    if valid and len(entities) == 3:
        return (text, {"entities": entities})
    return None

def generate_dataset(n=300):
    """Genera n ejemplos de entrenamiento variados."""
    dataset = []
    for _ in range(n * 3):  # intentos para alcanzar n válidos
        if len(dataset) >= n:
            break
        plantilla = random.choice(PLANTILLAS)
        producto = random.choice(PRODUCTOS)
        cantidad = random.choice(CANTIDADES)
        precio = random.choice(PRECIOS)
        example = generate_example(plantilla, producto, cantidad, precio)
        if example:
            dataset.append(example)
    return dataset

if __name__ == "__main__":
    data = generate_dataset(300)
    print(f"Ejemplos generados: {len(data)}")

    # Guardar como archivo Python importable
    with open("train_data.py", "w", encoding="utf-8") as f:
        f.write("# Datos de entrenamiento generados automáticamente\n")
        f.write("# Complementar con anotaciones reales de facturas Accemotos\n\n")
        f.write("TRAIN_DATA = [\n")
        for text, annots in data:
            f.write(f"    ({repr(text)}, {annots}),\n")
        f.write("]\n")

    print("Archivo train_data.py creado.")
    print("Primeros 3 ejemplos:")
    for ex in data[:3]:
        print(ex)
