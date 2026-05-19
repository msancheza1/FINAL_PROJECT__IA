"""
app/streamlit_app.py — Interfaz web del sistema de extracción de facturas.

Ejecutar con:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import tempfile
import os
import sys
import io

# Asegurar que el path incluya la raíz del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ocr import extract_text_from_pdf, extract_text_from_image, extract_text_from_text_file
from src.preprocess import clean_text, normalize_text, extract_candidate_lines
from src.predict import load_model, predict_entities, apply_hybrid_rules
from src.export import entities_to_dataframe, dataframe_to_csv, dataframe_to_json

# ─── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Extractor de Facturas IA",
    page_icon="📄",
    layout="wide",
)

# ─── Cache del modelo (se carga una sola vez) ────────────────────────────────
@st.cache_resource
def get_model():
    try:
        return load_model("models/modelo_facturas")
    except FileNotFoundError:
        return None

# ─── UI Principal ────────────────────────────────────────────────────────────
st.title("📄 Sistema Inteligente de Extracción de Facturas")
st.markdown("Sube una factura en PDF, imagen o texto y obtén el inventario en segundos.")

# Sidebar con información
with st.sidebar:
    st.header("ℹ️ Información")
    st.write("**Entidades detectadas:**")
    st.write("- 🟦 `PRODUCTO` — Nombre del artículo")
    st.write("- 🟩 `CANTIDAD` — Unidades")
    st.write("- 🟨 `PRECIO` — Valor unitario")
    st.divider()
    st.write("**Formatos soportados:**")
    st.write("PDF, PNG, JPG, TXT")

# Verificar que el modelo esté entrenado
nlp = get_model()
if nlp is None:
    st.error(
        "⚠️ Modelo no encontrado. Entrena el modelo primero:\n\n"
        "```bash\npython -m src.train_ner\n```"
    )
    st.stop()

# ─── Carga de archivo ────────────────────────────────────────────────────────
st.subheader("1️⃣ Sube la factura")
uploaded_file = st.file_uploader(
    "Arrastra aquí tu factura",
    type=["pdf", "png", "jpg", "jpeg", "txt"],
    label_visibility="collapsed",
)

# ─── También permitir texto directo (para demos) ─────────────────────────────
st.subheader("— o ingresa el texto directamente")
demo_text = st.text_area(
    "Pega el texto de la factura aquí:",
    height=120,
    placeholder="Ejemplo: CASCO SHAFT XTR-902 3 UND 120000\nGUANTES FOX 2 45000",
    label_visibility="collapsed",
)

# ─── Procesar ────────────────────────────────────────────────────────────────
if st.button("🚀 Extraer Inventario", type="primary", use_container_width=True):

    raw_text = ""

    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Extrayendo texto..."):
            if ext == "pdf":
                raw_text = extract_text_from_pdf(tmp_path)
            elif ext in ["png", "jpg", "jpeg"]:
                raw_text = extract_text_from_image(tmp_path)
            elif ext == "txt":
                raw_text = extract_text_from_text_file(tmp_path)

        os.unlink(tmp_path)

    elif demo_text.strip():
        raw_text = demo_text

    else:
        st.warning("Por favor sube un archivo o ingresa texto.")
        st.stop()

    # ─── Pipeline ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2️⃣ Texto extraído")
        cleaned = clean_text(raw_text)
        st.text_area("", value=cleaned, height=200, label_visibility="collapsed")

    # Predecir entidades
    with st.spinner("Analizando con IA..."):
        normalized = normalize_text(cleaned)
        entities = predict_entities(normalized, nlp=nlp)
        entities = apply_hybrid_rules(normalized, entities)

    with col2:
        st.subheader("3️⃣ Entidades detectadas")
        if entities:
            for ent in entities:
                color = {"PRODUCTO": "🟦", "CANTIDAD": "🟩", "PRECIO": "🟨"}.get(ent["label"], "⬜")
                st.write(f"{color} **{ent['label']}**: {ent['text']}")
        else:
            st.info("No se detectaron entidades. El modelo puede necesitar más entrenamiento.")

    # ─── Inventario ────────────────────────────────────────────────────────
    st.subheader("4️⃣ Inventario generado")
    df = entities_to_dataframe(entities)

    if df.empty:
        st.warning("No se pudo estructurar el inventario. Revisa las entidades detectadas.")
    else:
        st.dataframe(df, use_container_width=True)

        # ─── Descargas ─────────────────────────────────────────────────────
        col_csv, col_json = st.columns(2)

        with col_csv:
            # Usar BytesIO para crear un archivo CSV real
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            csv_buffer.seek(0)
            
            st.download_button(
                "⬇️ Descargar CSV",
                data=csv_buffer.getvalue(),
                file_name="inventario.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_json:
            import json
            json_data = json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Descargar JSON",
                data=json_data,
                file_name="inventario.json",
                mime="application/json",
                use_container_width=True,
            )

        st.success(f"✅ {len(df)} ítems detectados correctamente.")
