# app.py
import re, unicodedata
from io import BytesIO
from pathlib import Path
import joblib, pandas as pd, numpy as np, streamlit as st

st.set_page_config(page_title="Iowa House Prices - SVR", page_icon="🏠", layout="wide")

# ---------- Utils: normalizar encabezados ----------
def _norm_header(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _soft_key(s: str) -> str:
    s = _norm_header(s)
    s = s.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

# ---------- Carga del artifact (cache) ----------
@st.cache_resource
def load_artifact(path="modelo_svr.joblib"):
    art = joblib.load(path)
    pipe = art["pipeline_trained"]              # TTR(preprocess + SVR)
    req_cols = art["input_feature_names"]       # columnas que espera el modelo
    return art, pipe, req_cols

art, pipe, req_cols = load_artifact()

st.title("Predicción de precios de casas - Iowa (SVR)")
st.caption("Cargue un CSV/XLSX con las features; el modelo devolverá SalePrice_pred.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Acciones")
    # Descargar plantilla
    if st.button("⬇️ Descargar plantilla (CSV)"):
        import io
        buf = io.StringIO()
        pd.DataFrame(columns=req_cols).to_csv(buf, index=False)
        st.download_button("Descargar ahora", buf.getvalue().encode("utf-8-sig"),
                           file_name="plantilla_features.csv", mime="text/csv")
    st.info("Nota: si tu archivo trae la columna objetivo `SalePrice`, se ignorará.")

# ---------- Uploader ----------
st.subheader("1) Carga tus datos (CSV o Excel)")
up = st.file_uploader("Selecciona .csv o .xlsx", type=["csv","xlsx"])

def read_uploaded(f):
    if f is None: return None
    name = f.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(f)
        except Exception:
            f.seek(0)
            return pd.read_csv(f, encoding="latin-1", sep=None, engine="python")
    if name.endswith(".xlsx"):
        return pd.read_excel(f)
    return None

df_new = read_uploaded(up)

with st.expander("Ver columnas esperadas por el modelo"):
    st.write(req_cols)

if df_new is not None:
    # Normaliza encabezados
    df_new.columns = [_norm_header(c) for c in df_new.columns]
    st.subheader("2) Vista previa")
    st.dataframe(df_new.head(20), use_container_width=True)

    # Si viene SalePrice, quitarla (solo features)
    if "SalePrice" in df_new.columns:
        df_new = df_new.drop(columns=["SalePrice"])

    # Matching suave de columnas -> renombrar a nombres oficiales
    req_map  = {_soft_key(c): c for c in req_cols}        # clave suave -> oficial
    have_map = {_soft_key(c): c for c in df_new.columns}  # clave suave -> actual
    rename_map = {have_map[k]: req_map[k] for k in req_map.keys() if k in have_map}
    df_new = df_new.rename(columns=rename_map)

    faltan_soft = [k for k in req_map.keys() if k not in have_map]
    extra_soft  = [k for k in have_map.keys() if k not in req_map]
    faltan = [req_map[k] for k in faltan_soft]
    extra  = [have_map[k] for k in extra_soft]

    c1, c2 = st.columns(2)
    with c1:
        if extra: st.warning(f"Columnas extra: {extra} (se ignorarán)")
        else:     st.success("No hay columnas extra.")
    with c2:
        if faltan: st.info(f"Faltan columnas requeridas: {faltan}. "
                           "Se crearán como NaN y serán imputadas por el pipeline.")
        else:       st.success("No faltan columnas requeridas.")

    st.subheader("3) Ejecutar predicción")
    if st.button("Predecir", type="primary"):
        # Alinear esquema: añadir faltantes como NaN y ordenar columnas
        for c in req_cols:
            if c not in df_new.columns:
                df_new[c] = np.nan
        X = df_new.reindex(columns=req_cols)

        # Predecir (TTR devuelve ya en escala original)
        yhat = pipe.predict(X)

        # Salida: SOLO features del modelo + predicción
        out = X.copy()
        out["SalePrice_pred"] = yhat

        st.success("¡Predicción completada!")
        st.subheader("4) Resultados (muestra)")
        st.dataframe(out.head(50), use_container_width=True)

        # Descargas
        def to_csv_bytes(df): return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        def to_excel_bytes(df):
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as w:
                df.to_excel(w, index=False, sheet_name="predicciones")
            return bio.getvalue()

        st.download_button("⬇️ Descargar CSV", to_csv_bytes(out), "predicciones_svr.csv", "text/csv")
        st.download_button("⬇️ Descargar Excel", to_excel_bytes(out),
                           "predicciones_svr.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Sube un archivo para comenzar.")
