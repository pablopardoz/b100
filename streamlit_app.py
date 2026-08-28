import datetime
import hashlib
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import Client, create_client

load_dotenv()

st.set_page_config(page_title="Sincronizador Financiero", layout="wide")


# =====================================================================
# INICIALIZACIÓN DE CLIENTES EXTERNOS
# =====================================================================
def leer_secreto(clave: str) -> str:
    """Devuelve el secreto desde st.secrets y, si no está, desde el entorno / .env."""
    try:
        valor = st.secrets[clave]
    except Exception:
        valor = None
    if not valor:
        valor = os.getenv(clave)
    if not valor or not str(valor).strip():
        st.error(
            f"Falta el secreto `{clave}`. Rellénalo en `.env` o en "
            "`.streamlit/secrets.toml` y recarga la app."
        )
        st.stop()
    return str(valor).strip()


@st.cache_resource
def init_supabase() -> Client:
    return create_client(leer_secreto("SUPABASE_URL"), leer_secreto("SUPABASE_KEY"))


@st.cache_resource
def init_gemini() -> genai.Client:
    return genai.Client(api_key=leer_secreto("GEMINI_API_KEY"))


supabase_client = init_supabase()
gemini_client = init_gemini()


# =====================================================================
# LÓGICA DE NEGOCIO
# =====================================================================
def generar_hash_fila(row: pd.Series) -> str:
    valores_concatenados = "".join(str(val).strip().lower() for val in row.values)
    return hashlib.sha256(valores_concatenados.encode('utf-8')).hexdigest()


# Fuente única de verdad: cada categoría lleva el color que usa el gráfico y un
# cuadrado/círculo del mismo tono que sirve de pista visual en el desplegable.
PALETA_CATEGORIAS: dict[str, tuple[str, str]] = {
    "SUPERMERCADOS": ("#43A047", "🟩"),
    "RESTAURACION": ("#FB8C00", "🟧"),
    "FACTURAS": ("#1E88E5", "🟦"),
    "HOGAR": ("#8D6E63", "🟫"),
    "SALUD": ("#E53935", "🟥"),
    "HIJAS": ("#8E24AA", "🟪"),
    "MASCOTAS": ("#EF5350", "🔴"),
    "TRANSPORTE": ("#FDD835", "🟨"),
    "VIAJES": ("#26C6DA", "🩵"),
    "OCIO": ("#5C6BC0", "🔵"),
    "INGRESOS": ("#00897B", "🟢"),
    "AHORRO": ("#546E7A", "⚫"),
    "OTROS": ("#BDBDBD", "⬜"),
}

# Categorías antiguas que ya no existen y se muestran unificadas en otra.
ALIAS_CATEGORIAS = {"FARMACIA": "SALUD"}

CATEGORIAS = list(PALETA_CATEGORIAS)
COLOR_POR_CATEGORIA = {nombre: color for nombre, (color, _) in PALETA_CATEGORIAS.items()}
ETIQUETA_POR_CATEGORIA = {
    nombre: f"{emoji} {nombre}" for nombre, (_, emoji) in PALETA_CATEGORIAS.items()
}
CATEGORIA_POR_ETIQUETA = {v: k for k, v in ETIQUETA_POR_CATEGORIA.items()}
ETIQUETAS_CATEGORIAS = list(CATEGORIA_POR_ETIQUETA)
ETIQUETA_OTROS = ETIQUETA_POR_CATEGORIA["OTROS"]


def a_etiqueta(categoria) -> str:
    """Nombre guardado en Supabase -> etiqueta con color para el desplegable."""
    nombre = str(categoria).strip().upper()
    nombre = ALIAS_CATEGORIAS.get(nombre, nombre)
    return ETIQUETA_POR_CATEGORIA.get(nombre, ETIQUETA_OTROS)


def a_categoria(etiqueta) -> str:
    """Etiqueta del desplegable -> nombre que se guarda en Supabase."""
    return CATEGORIA_POR_ETIQUETA.get(str(etiqueta).strip(), "OTROS")

TAMANO_LOTE = 50

# Tope del slider de fechas: 3 meses => como mucho 90 barras en el gráfico diario.
DIAS_MAX_RANGO = 90

PROMPT_CATEGORIZACION = f"""\
Eres un asistente experto en finanzas personales. Tu única tarea es clasificar
descripciones de movimientos bancarios.

Recibirás una lista numerada de conceptos. Asigna a cada uno exactamente una de estas
categorías: {", ".join(CATEGORIAS)}.

Devuelve un array JSON de strings con una categoría por concepto, en el mismo orden que
los recibiste y con exactamente el mismo número de elementos que conceptos de entrada.
Pistas: HIJAS es para colegio, actividades y gastos de las niñas; FACTURAS para luz,
agua, gas, teléfono y suscripciones recurrentes; AHORRO para traspasos a inversión.
Si un concepto no encaja con claridad en ninguna categoría, usa OTROS.
"""


def _categorizar_lote(conceptos: list[str]) -> list[str]:
    """Clasifica un lote de conceptos. Devuelve una lista del mismo tamaño."""
    contenido_entrada = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(conceptos))

    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=contenido_entrada,
        config=types.GenerateContentConfig(
            system_instruction=PROMPT_CATEGORIZACION,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING, enum=CATEGORIAS),
            ),
        ),
    )

    if not response.text:
        raise ValueError("La respuesta del modelo llegó vacía")

    resultado = json.loads(response.text)
    if not isinstance(resultado, list):
        raise ValueError(f"Se esperaba un array JSON, llegó {type(resultado).__name__}")

    categorias = [
        str(c).strip().upper() if str(c).strip().upper() in CATEGORIAS else "OTROS"
        for c in resultado
    ]

    if len(categorias) != len(conceptos):
        st.warning(
            f"El modelo devolvió {len(categorias)} categorías para {len(conceptos)} "
            "conceptos; los que falten quedan como OTROS."
        )
        categorias = categorias[: len(conceptos)]
        categorias += ["OTROS"] * (len(conceptos) - len(categorias))

    return categorias


def categorizar_conceptos_con_llm(conceptos: list[str]) -> list[str]:
    """Clasifica los conceptos en lotes. Siempre devuelve len(conceptos) categorías."""
    if not conceptos:
        return []

    categorias: list[str] = []
    for inicio in range(0, len(conceptos), TAMANO_LOTE):
        lote = conceptos[inicio : inicio + TAMANO_LOTE]
        try:
            categorias.extend(_categorizar_lote(lote))
        except Exception as e:
            st.error(
                f"Error categorizando los conceptos {inicio + 1}-{inicio + len(lote)}: {e}"
            )
            categorias.extend(["OTROS"] * len(lote))

    return categorias


def consultar_movimientos_sin_categorizar() -> pd.DataFrame:
    """Devuelve los movimientos ya guardados que quedaron como OTROS."""
    try:
        res = (
            supabase_client.table("movimientos_cuenta")
            .select("row_hash, concepto")
            .eq("categoria", "OTROS")
            .execute()
        )
        if res.data:
            return pd.DataFrame(res.data)
    except Exception as e:
        st.error(f"Error al consultar los movimientos sin categorizar: {e}")
    return pd.DataFrame()


def actualizar_categorias_en_lote(asignaciones: dict[str, str]) -> int:
    """Guarda {row_hash: categoria} agrupando por categoría.

    Hace una sola petición por categoría distinta (no una por fila), así que guardar
    5 o 500 movimientos cuesta como mucho tantas llamadas como categorías haya.
    Devuelve cuántas filas se han actualizado.
    """
    if not asignaciones:
        return 0

    por_categoria: dict[str, list[str]] = {}
    for row_hash, categoria in asignaciones.items():
        por_categoria.setdefault(categoria, []).append(row_hash)

    actualizadas = 0
    for categoria, hashes in por_categoria.items():
        try:
            (
                supabase_client.table("movimientos_cuenta")
                .update({"categoria": categoria})
                .in_("row_hash", hashes)
                .execute()
            )
            actualizadas += len(hashes)
        except Exception as e:
            st.error(f"Error al guardar {len(hashes)} movimiento(s) como {categoria}: {e}")

    return actualizadas


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
                        st.warning(
                            "La categorización devolvió "
                            f"{len(categorias_asignadas)} valores para "
                            f"{len(df_nuevos)} filas: se guardan como OTROS."
                        )
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

    st.divider()
    st.caption(
        "Vuelve a pasar por Gemini los movimientos que ya están guardados "
        "con la categoría OTROS."
    )

    if st.button("🏷️ Recategorizar OTROS", use_container_width=True):
        with st.status("Buscando movimientos sin categorizar...", expanded=True) as status:
            df_otros = consultar_movimientos_sin_categorizar()

            if df_otros.empty:
                status.update(
                    label="No hay movimientos en OTROS que recategorizar.",
                    state="complete",
                )
            else:
                total_otros = len(df_otros)
                status.update(label=f"Recategorizando {total_otros} movimientos...")

                conceptos_otros = df_otros['concepto'].astype(str).tolist()
                categorias_nuevas = categorizar_conceptos_con_llm(conceptos_otros)

                if len(categorias_nuevas) != total_otros:
                    status.update(
                        label="La categorización no devolvió un valor por movimiento; "
                        "no se ha modificado nada.",
                        state="error",
                    )
                else:
                    asignaciones = {
                        str(row_hash): categoria
                        for row_hash, categoria in zip(
                            df_otros['row_hash'], categorias_nuevas
                        )
                        if categoria != "OTROS"
                    }
                    actualizados = actualizar_categorias_en_lote(asignaciones)

                    if actualizados:
                        status.update(
                            label=f"{actualizados} de {total_otros} movimientos "
                            "recategorizados.",
                            state="complete",
                        )
                        st.rerun()
                    else:
                        status.update(
                            label="Gemini mantuvo todos los movimientos en OTROS.",
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

    if fecha_min == fecha_max:
        desde = hasta = fecha_min
        st.caption(f"Solo hay movimientos del {fecha_min.strftime('%d/%m/%Y')}.")
    else:
        # El slider arranca mostrando la ventana máxima más reciente. Arrastrando
        # cualquiera de los dos extremos se mueve el rango día a día.
        arranque_desde = max(
            fecha_min, fecha_max - datetime.timedelta(days=DIAS_MAX_RANGO - 1)
        )
        desde, hasta = st.slider(
            "Rango de fechas",
            min_value=fecha_min,
            max_value=fecha_max,
            value=(arranque_desde, fecha_max),
            format="DD/MM/YYYY",
            key="rango_fechas",
        )

        dias_seleccionados = (hasta - desde).days + 1
        if dias_seleccionados > DIAS_MAX_RANGO:
            hasta = desde + datetime.timedelta(days=DIAS_MAX_RANGO - 1)
            st.warning(
                f"El rango máximo es de {DIAS_MAX_RANGO} días (3 meses). Se recorta "
                f"hasta el {hasta.strftime('%d/%m/%Y')}; mueve el extremo izquierdo "
                "para avanzar en el tiempo."
            )
            dias_seleccionados = DIAS_MAX_RANGO

        st.caption(
            f"{desde.strftime('%d/%m/%Y')} → {hasta.strftime('%d/%m/%Y')} "
            f"· {dias_seleccionados} día(s)"
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

    # Los KPIs y el gráfico se pintan arriba, pero necesitan lo que el usuario marque
    # en la tabla, que va abajo. Reservando los contenedores en el orden visual
    # deseado podemos rellenarlos después, en el orden que exige la lógica.
    contenedor_kpis = st.container()
    contenedor_listado = st.container()

    df_flujo = df_flujo.reset_index(drop=True)
    df_flujo.insert(0, 'incluir', True)
    df_flujo['categoria'] = df_flujo['categoria'].map(a_etiqueta)

    # st.data_editor guarda las ediciones por POSICIÓN de fila (edited_rows es un
    # dict[int, ...]). Si el key no cambia, esas posiciones se re-aplican al nuevo
    # conjunto de filas cuando cambias de fecha o de flujo, y la edición aterriza
    # en un movimiento distinto. Atando el key a la identidad de las filas visibles,
    # cambiar de filtro crea un editor nuevo y limpio.
    firma_filas = hashlib.sha256(
        "|".join(df_flujo['row_hash'].astype(str)).encode('utf-8')
    ).hexdigest()[:12]
    key_editor = f"editor_movimientos_{firma_filas}"

    # =================================================================
    # LISTADO EDITABLE (plegado por defecto)
    # =================================================================
    with contenedor_listado:
        with st.expander(
            f"📋 Movimientos y categorías ({len(df_flujo)})", expanded=False
        ):
            st.caption(
                "Desmarca un movimiento para excluirlo de los cálculos y corrige su "
                "categoría en el desplegable. Los cambios de categoría no se guardan "
                "hasta que pulses **Guardar categorías**."
            )

            # row_hash viaja en el DataFrame para poder identificar cada fila al
            # guardar, pero se oculta de la vista con column_order.
            df_tabla_interactiva = st.data_editor(
                df_flujo[
                    ['incluir', 'fecha', 'concepto', 'importe', 'categoria', 'row_hash']
                ],
                use_container_width=True,
                hide_index=True,
                key=key_editor,
                column_order=('incluir', 'fecha', 'concepto', 'importe', 'categoria'),
                column_config={
                    "incluir": st.column_config.CheckboxColumn(
                        "¿Incluir?", default=True
                    ),
                    "fecha": st.column_config.DateColumn(
                        "Fecha", format="DD/MM/YYYY", disabled=True
                    ),
                    "concepto": st.column_config.TextColumn(
                        "Concepto", disabled=True
                    ),
                    "importe": st.column_config.NumberColumn(
                        "Importe (€)", format="%.2f", disabled=True
                    ),
                    "categoria": st.column_config.SelectboxColumn(
                        "Categoría",
                        options=ETIQUETAS_CATEGORIAS,
                        required=True,
                        width="medium",
                    ),
                },
            )

            # --- Persistencia manual de las categorías editadas ---
            # La comparación va por row_hash contra lo que hay en Supabase, no por
            # posición de fila, así que solo se escribe lo que has tocado de verdad.
            categoria_en_bd = {
                str(h): a_categoria(a_etiqueta(c))
                for h, c in zip(df_base['row_hash'], df_base['categoria'])
            }

            hashes_tabla = df_tabla_interactiva['row_hash'].astype(str)
            categorias_ahora = df_tabla_interactiva['categoria'].map(a_categoria)
            categorias_antes = hashes_tabla.map(categoria_en_bd)
            filas_cambiadas = categorias_ahora != categorias_antes

            df_cambios = pd.DataFrame(
                {
                    "row_hash": hashes_tabla[filas_cambiadas],
                    "concepto": df_tabla_interactiva.loc[filas_cambiadas, 'concepto'],
                    "antes": categorias_antes[filas_cambiadas].fillna("?"),
                    "ahora": categorias_ahora[filas_cambiadas],
                }
            )
            total_cambios = len(df_cambios)

            col_g1, col_g2 = st.columns([1, 3])
            with col_g1:
                pulsado_guardar = st.button(
                    "💾 Guardar categorías",
                    use_container_width=True,
                    disabled=total_cambios == 0,
                )
            with col_g2:
                if total_cambios:
                    st.warning(
                        f"{total_cambios} categoría(s) sin guardar. Cambia todas las "
                        "que quieras y guárdalas de una vez."
                    )
                else:
                    st.caption("Sin cambios pendientes.")

            if total_cambios:
                st.dataframe(
                    df_cambios[['concepto', 'antes', 'ahora']],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "concepto": st.column_config.Column("Se va a guardar"),
                        "antes": st.column_config.Column("Antes"),
                        "ahora": st.column_config.Column("Ahora"),
                    },
                )

            if pulsado_guardar:
                asignaciones = dict(zip(df_cambios['row_hash'], df_cambios['ahora']))
                guardadas = actualizar_categorias_en_lote(asignaciones)
                if guardadas:
                    # Se limpia el estado del editor para que ninguna edición ya
                    # guardada quede colgando y se re-aplique en el siguiente rerun.
                    st.session_state.pop(key_editor, None)
                    st.success(f"{guardadas} de {total_cambios} categorías guardadas.")
                    st.rerun()

    df_final_calculos = df_tabla_interactiva[df_tabla_interactiva['incluir']].copy()
    df_final_calculos['categoria'] = df_final_calculos['categoria'].map(a_categoria)

    if "Gastos" in tipo_flujo:
        df_final_calculos['importe_abs'] = df_final_calculos['importe'].abs()
    else:
        df_final_calculos['importe_abs'] = df_final_calculos['importe']

    # =================================================================
    # KPIs Y GRÁFICO (arriba)
    # =================================================================
    with contenedor_kpis:
        if df_final_calculos.empty:
            st.info(
                "No hay registros activos marcados o no existen transacciones "
                "en este rango de fechas para el flujo seleccionado."
            )
        else:
            total_operaciones = len(df_final_calculos)
            suma_total = df_final_calculos['importe'].sum()
            media_operacion = df_final_calculos['importe_abs'].mean()

            df_resumen_cat = (
                df_final_calculos
                .groupby('categoria')['importe_abs']
                .sum()
                .reset_index()
                .sort_values(by='importe_abs', ascending=False)
            )
            categoria_top = df_resumen_cat.iloc[0]

            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
            col_k1.metric("Movimientos", total_operaciones)
            col_k2.metric("Impacto Neto", f"{suma_total:,.2f} €")
            col_k3.metric("Media por Movimiento", f"{media_operacion:,.2f} €")
            col_k4.metric(
                f"Top: {categoria_top['categoria'].title()}",
                f"{categoria_top['importe_abs']:,.2f} €",
            )

            st.divider()

            # --- Gasto por día: una barra por cada día del rango del slider ---
            df_por_dia = (
                df_final_calculos
                .groupby(['fecha', 'categoria'])['importe_abs']
                .sum()
                .reset_index()
            )
            etiqueta_flujo = "Gasto" if "Gastos" in tipo_flujo else "Ingreso"

            fig_dias = px.bar(
                df_por_dia,
                x='fecha',
                y='importe_abs',
                color='categoria',
                title=f"{etiqueta_flujo} por día",
                color_discrete_map=COLOR_POR_CATEGORIA,
                category_orders={'categoria': CATEGORIAS},
                labels={
                    'fecha': '',
                    'importe_abs': f'{etiqueta_flujo} (€)',
                    'categoria': 'Categoría',
                },
            )
            fig_dias.update_layout(
                barmode='stack',
                bargap=0.15,
                margin=dict(t=40, b=0, l=0, r=0),
                legend=dict(
                    orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0
                ),
            )
            # El eje abarca siempre todo el rango del slider, aunque haya días sin
            # movimientos: así el ancho de barra no cambia al mover el slider.
            fig_dias.update_xaxes(
                range=[
                    desde - datetime.timedelta(days=1),
                    hasta + datetime.timedelta(days=1),
                ],
                dtick=86400000.0 * max(1, (hasta - desde).days // 15 + 1),
                tickformat='%d/%m',
            )
            fig_dias.update_yaxes(ticksuffix=' €')
            st.plotly_chart(fig_dias, use_container_width=True)

            st.divider()

            col_m1, col_m2 = st.columns([2, 3])

            with col_m1:
                st.subheader("📊 Por Categoría")
                estilo_resumen = df_resumen_cat.style.apply(
                    lambda columna: [
                        f"background-color: {COLOR_POR_CATEGORIA.get(v, '#BDBDBD')};"
                        " color: white"
                        for v in columna
                    ],
                    subset=['categoria'],
                ).format({'importe_abs': '{:,.2f} €'})
                st.dataframe(
                    estilo_resumen,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "categoria": st.column_config.Column("Categoría"),
                        "importe_abs": st.column_config.Column("Total"),
                    },
                )

            with col_m2:
                fig = px.pie(
                    df_final_calculos,
                    values='importe_abs',
                    names='categoria',
                    title=titulo_grafico,
                    hole=0.4,
                    color='categoria',
                    color_discrete_map=COLOR_POR_CATEGORIA,
                    category_orders={'categoria': CATEGORIAS},
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
else:
    st.info("La base de datos está vacía. Sube tus CSV desde el panel izquierdo.")
