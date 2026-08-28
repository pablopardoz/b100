import streamlit as st
import pandas as pd
import hashlib
import datetime
import plotly.express as px
from supabase import create_client, Client
from google import genai
from google.genai import types

st.set_page_config(page_title="Sincronizador Financiero", layout="wide")


# =====================================================================
# INICIALIZACIÓN DE CLIENTES EXTERNOS
# =====================================================================
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"].strip()
    return create_client(url, st.secrets["SUPABASE_KEY"])


@st.cache_resource
def init_gemini() -> genai.Client:
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


supabase_client = init_supabase()
gemini_client = init_gemini()


# =====================================================================
# LÓGICA DE NEGOCIO
# =====================================================================
def generar_hash_fila(row: pd.Series) -> str:
    valores_concatenados = "".join(str(val).strip().lower() for val in row.values)
    return hashlib.sha256(valores_concatenados.encode('utf-8')).hexdigest()


def categorizar_conceptos_con_llm(conceptos: list[str]) -> list[str]:
    if not conceptos:
        return []

    categorias_fijas = "SUPERMERCADOS, RESTAURACION, FARMACIA, SALUD, HOGAR, FACTURAS, OTROS"
    prompt_sistema = f"""
    Eres un asistente experto en finanzas personales. Tu única tarea es clasificar descripciones de movimientos bancarios.
    Debes mapear cada concepto provisto a una de las siguientes categorías exactas: [{categorias_fijas}].
    Responde estrictamente con la lista de categorías separadas por saltos de línea, en el mismo orden recibido.
    """
    contenido_entrada = "\n".join(f"- {c}" for c in conceptos)

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contenido_entrada,
            config=types.GenerateContentConfig(
                system_instruction=prompt_sistema,
                temperature=0.1,
            ),
        )
        lineas = [
            line.strip().upper()
            for line in response.text.strip().split("\n")
            if line.strip()
        ]
        if len(lineas) != len(conceptos):
            return [
                lineas[i] if i < len(lineas) else "OTROS"
                for i in range(len(conceptos))
            ]
        return lineas
    except Exception as e:
        st.error(f"Error procesando la categorización con el LLM: {e}")
        return ["OTROS"] * len(conceptos)


def consultar_datos_existentes() -> pd.DataFrame:
    try:
        res = supabase_client.table("movimientos_cuenta").select("*").execute()
        if res.data:
            return pd.DataFrame(res.data)
    except Exception as e:
        st.error(f"Error al consultar Supabase: {e}")
    return pd.DataFrame()


def limpiar_importe(val_importe) -> float:
    try:
        if isinstance(val_importe, str):
            val_limpio = val_importe.replace('€', '').strip()
            if ',' in val_limpio and '.' in val_limpio:
                val_limpio = val_limpio.replace('.', '').replace(',', '.')
            elif ',' in val_limpio:
                val_limpio = val_limpio.replace(',', '.')
            return float(val_limpio)
        return float(val_importe)
    except Exception:
        return 0.0


def limpiar_fecha(val_fecha) -> str:
    hoy = datetime.date.today().strftime('%Y-%m-%d')
    try:
        fecha = pd.to_datetime(str(val_fecha).strip(), dayfirst=True, errors='coerce')
        if pd.isna(fecha):
            return hoy
        return fecha.strftime('%Y-%m-%d')
    except Exception:
        return hoy


# =====================================================================
# INTERFAZ DE USUARIO - PANEL LATERAL (SINCRONIZACIÓN)
# =====================================================================
with st.sidebar:
    st.header("Subida de Archivos")
    st.write("Arrastra tus archivos CSV bancarios.")

    archivos_subidos = st.file_uploader(
        "Selecciona uno o varios archivos CSV",
        type=["csv"],
        accept_multiple_files=True,
    )
    boton_deshabilitado = not bool(archivos_subidos)

    if st.button(
        "🔄 Procesar y Sincronizar",
        use_container_width=True,
        disabled=boton_deshabilitado,
    ):
        with st.status("Procesando archivos locales...", expanded=True) as status:
            listado_dfs = []
            for archivo in archivos_subidos:
                try:
                    df_individual = pd.read_csv(archivo)
                    if not df_individual.empty:
                        listado_dfs.append(df_individual)
                except Exception as e:
                    st.sidebar.error(f"Error al leer {archivo.name}: {e}")

            if not listado_dfs:
                status.update(
                    label="Sincronización detenida: Archivos vacíos.",
                    state="error",
                )
            else:
                df_total = pd.concat(listado_dfs, ignore_index=True)
                df_total.columns = [col.lower().strip() for col in df_total.columns]

                status.update(label="Calculando huellas digitales únicas...")
                df_total['row_hash'] = df_total.apply(generar_hash_fila, axis=1)
                df_unico = df_total.drop_duplicates(subset=['row_hash'])

                status.update(label="Consultando deltas en base de datos...")
                df_actual_db = consultar_datos_existentes()

                if not df_actual_db.empty:
                    hashes_existentes = set(df_actual_db['row_hash'].tolist())
                    df_nuevos = df_unico[
                        ~df_unico['row_hash'].isin(hashes_existentes)
                    ].copy()
                else:
                    df_nuevos = df_unico.copy()

                cantidad_nuevos = len(df_nuevos)
                st.write(
                    f"Filas únicas leídas: **{len(df_unico)}**. "
                    f"Registros nuevos: **{cantidad_nuevos}**."
                )

                if cantidad_nuevos > 0:
                    col_fecha = 'fecha de operación'
                    col_concepto = 'concepto'
                    col_importe = 'cantidad'

                    columnas = list(df_nuevos.columns)
                    if col_fecha not in columnas:
                        col_fecha = columnas[0]
                    if col_concepto not in columnas:
                        col_concepto = columnas[1]
                    if col_importe not in columnas:
                        col_importe = columnas[2]

                    status.update(
                        label=f"Categorizando {cantidad_nuevos} transacciones con Gemini..."
                    )
                    conceptos_lista = df_nuevos[col_concepto].astype(str).tolist()
                    categorias_asignadas = categorizar_conceptos_con_llm(conceptos_lista)

                    if len(categorias_asignadas) == len(df_nuevos):
                        df_nuevos['categoria'] = categorias_asignadas
                    else:
                        df_nuevos['categoria'] = ["OTROS"] * len(df_nuevos)

                    status.update(label="Insertando registros en Supabase...")
                    registros_a_subir = [
                        {
                            "row_hash": str(fila['row_hash']),
                            "fecha": limpiar_fecha(fila[col_fecha]),
                            "concepto": str(fila[col_concepto]),
                            "importe": limpiar_importe(fila[col_importe]),
                            "categoria": str(fila['categoria']),
                        }
                        for _, fila in df_nuevos.iterrows()
                    ]

                    try:
                        supabase_client.table("movimientos_cuenta").insert(
                            registros_a_subir
                        ).execute()
                        status.update(
                            label=f"¡Éxito! {cantidad_nuevos} transacciones guardadas.",
                            state="complete",
                        )
                        st.rerun()
                    except Exception as ins_err:
                        status.update(
                            label=f"Error al escribir datos: {ins_err}",
                            state="error",
                        )
                else:
                    status.update(
                        label="Los movimientos ya están al día.",
                        state="complete",
                    )

# =====================================================================
# CUERPO PRINCIPAL DEL DASHBOARD
# =====================================================================
st.title("📊 Cuadro de Mando Financiero")

df_base = consultar_datos_existentes()

if not df_base.empty:
    df_base['fecha'] = pd.to_datetime(df_base['fecha'], errors='coerce').dt.date
    df_base['importe'] = pd.to_numeric(df_base['importe'], errors='coerce').fillna(0.0)
    df_base = df_base.sort_values(by='fecha', ascending=False).reset_index(drop=True)

    fecha_min = df_base['fecha'].min()
    fecha_max = df_base['fecha'].max()

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        desde = st.date_input(
            "Desde:", value=fecha_min, min_value=fecha_min, max_value=fecha_max
        )
    with col_f2:
        hasta = st.date_input(
            "Hasta:", value=fecha_max, min_value=fecha_min, max_value=fecha_max
        )

    st.divider()

    df_filtrado_fecha = df_base[
        (df_base['fecha'] >= desde) & (df_base['fecha'] <= hasta)
    ].copy()

    tipo_flujo = st.radio(
        "Tipo de Transacciones a analizar:",
        options=["Gastos (Importes negativos)", "Ingresos (Importes positivos)"],
        horizontal=True,
    )

    if "Gastos" in tipo_flujo:
        df_flujo = df_filtrado_fecha[df_filtrado_fecha['importe'] < 0].copy()
        df_flujo['importe_abs'] = df_flujo['importe'].abs()
        titulo_grafico = "Distribución de Gastos por Categoría"
    else:
        df_flujo = df_filtrado_fecha[df_filtrado_fecha['importe'] >= 0].copy()
        df_flujo['importe_abs'] = df_flujo['importe']
        titulo_grafico = "Distribución de Ingresos por Categoría"

    st.subheader("📋 Selección de Movimientos Activos")
    st.write("Desmarca transacciones individuales en la casilla para recalcular:")

    df_flujo = df_flujo.reset_index(drop=True)
    df_flujo.insert(0, 'incluir', True)

    df_tabla_interactiva = st.data_editor(
        df_flujo[['incluir', 'fecha', 'concepto', 'importe', 'categoria']],
        use_container_width=True,
        hide_index=True,
        key="editor_movimientos",
        column_config={
            "incluir": st.column_config.CheckboxColumn("¿Incluir?", default=True),
            "fecha": st.column_config.DateColumn(
                "Fecha", format="DD/MM/YYYY", disabled=True
            ),
            "concepto": st.column_config.TextColumn("Concepto", disabled=True),
            "importe": st.column_config.NumberColumn(
                "Importe (€)", format="%.2f", disabled=True
            ),
            "categoria": st.column_config.TextColumn("Categoría", disabled=True),
        },
    )

    df_final_calculos = df_tabla_interactiva[df_tabla_interactiva['incluir']].copy()

    if "Gastos" in tipo_flujo:
        df_final_calculos['importe_abs'] = df_final_calculos['importe'].abs()
    else:
        df_final_calculos['importe_abs'] = df_final_calculos['importe']

    st.divider()

    if not df_final_calculos.empty:
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.subheader("📊 Resumen")
            total_operaciones = len(df_final_calculos)
            suma_total = df_final_calculos['importe'].sum()
            st.metric("Movimientos Seleccionados", total_operaciones)
            st.metric("Movimientos Seleccionados", total_operaciones)
            st.metric("Impacto Neto Seleccionado", f"{suma_total:,.2f} €")
            
            df_resumen_cat = (
                df_final_calculos
                .groupby('categoria')['importe_abs']
                .sum()
                .reset_index()
                .sort_values(by='importe_abs', ascending=False)
            )
            st.dataframe(df_resumen_cat, hide_index=True, use_container_width=True)

        with col_m2:
            fig = px.pie(
                df_final_calculos,
                values='importe_abs',
                names='categoria',
                title=titulo_grafico,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No hay registros activos marcados o no existen transacciones "
            "en este rango de fechas para el flujo seleccionado."
        )
else:
    st.info("La base de datos está vacía. Sube tus CSV desde el panel izquierdo.")
