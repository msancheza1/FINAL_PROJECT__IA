"""
src/train_ner.py — Entrenamiento del modelo NER con spaCy.

Uso:
    python -m src.train_ner

El modelo entrenado se guarda en models/modelo_facturas/
"""

import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import sys
import os

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.annotations.train_data import TRAIN_DATA


def train_ner_model(
    train_data: list,
    output_dir: str = "models/modelo_facturas",
    n_iter: int = 50,
    dropout: float = 0.3,
):
    """
    Entrena un modelo NER desde cero con spaCy en español.

    Args:
        train_data: Lista de (texto, {"entities": [...]})
        output_dir: Ruta donde guardar el modelo
        n_iter: Número de épocas
        dropout: Tasa de dropout para regularización
    """
    print(f"Iniciando entrenamiento con {len(train_data)} ejemplos...")

    # Crear modelo en blanco en español
    nlp = spacy.blank("es")

    # Agregar pipe de NER
    ner = nlp.add_pipe("ner", last=True)

    # Registrar etiquetas
    labels = ["PRODUCTO", "CANTIDAD", "PRECIO"]
    for label in labels:
        ner.add_label(label)

    # Deshabilitar otros pipes durante el entrenamiento
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        best_loss = float("inf")

        for epoch in range(n_iter):
            random.shuffle(train_data)
            losses = {}

            # Mini-batches con tamaño variable (mejora generalización)
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    try:
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    except Exception as e:
                        # Saltar ejemplos con spans inválidos
                        continue

                if examples:
                    nlp.update(
                        examples,
                        drop=dropout,
                        losses=losses,
                    )

            ner_loss = losses.get("ner", 0)

            if epoch % 10 == 0 or epoch == n_iter - 1:
                print(f"Época {epoch + 1}/{n_iter} — Loss NER: {ner_loss:.4f}")

            # Guardar mejor modelo
            if ner_loss < best_loss:
                best_loss = ner_loss

    # Guardar modelo final
    os.makedirs(output_dir, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"\n✅ Modelo guardado en: {output_dir}")
    print(f"   Loss final: {best_loss:.4f}")

    return nlp


def evaluate_model(nlp, test_data: list):
    """
    Evalúa el modelo con métricas de NER: Precision, Recall, F1.

    Args:
        nlp: Modelo spaCy cargado
        test_data: Lista de (texto, {"entities": [...]})
    """
    from spacy.scorer import Scorer

    scorer = Scorer()
    examples = []

    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        try:
            example = Example.from_dict(doc, annotations)
            # Aplicar predicción
            pred_doc = nlp(text)
            example.predicted = pred_doc
            examples.append(example)
        except Exception:
            continue

    scores = scorer.score(examples)

    print("\n📊 Resultados de Evaluación:")
    print(f"  Precision:  {scores['ents_p']:.3f}")
    print(f"  Recall:     {scores['ents_r']:.3f}")
    print(f"  F1-Score:   {scores['ents_f']:.3f}")
    print("\n  Por entidad:")
    for ent_type, ent_scores in scores.get("ents_per_type", {}).items():
        print(f"    {ent_type}: P={ent_scores['p']:.3f}  R={ent_scores['r']:.3f}  F={ent_scores['f']:.3f}")

    return scores


if __name__ == "__main__":
    # Separar 80% train / 20% test
    random.shuffle(TRAIN_DATA)
    split = int(len(TRAIN_DATA) * 0.8)
    train_set = TRAIN_DATA[:split]
    test_set = TRAIN_DATA[split:]

    print(f"Train: {len(train_set)} ejemplos | Test: {len(test_set)} ejemplos")

    # Entrenar
    nlp = train_ner_model(train_set, n_iter=50)

    # Evaluar
    evaluate_model(nlp, test_set)
