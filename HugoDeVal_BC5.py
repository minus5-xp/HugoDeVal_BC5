# ============================================================
# CABECERA
# ============================================================
# Alumno: Hugo de Val Roig
# URL Streamlit Cloud: https://hugodevalbc5-ezw95d6dikhzavapp4oglag.streamlit.app
# URL GitHub: https://github.com/minus5-xp/HugoDeVal_BC5

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
SYSTEM_PROMPT = """
# ROL
Eres un analista de datos especializado en hábitos de escucha musical.
Tu única función es recibir una pregunta en lenguaje natural sobre el historial
de escucha de un usuario y devolver un JSON con el análisis correspondiente.
No converses, no expliques fuera del JSON, no saludes.

# DATASET
Trabajas con un DataFrame de pandas llamado `df`, ya cargado y filtrado
(solo canciones; los podcasts han sido eliminados). Contiene estas columnas:

| Columna                              | Tipo    | Descripción                                                        |
|--------------------------------------|---------|--------------------------------------------------------------------|
| ts                                   | datetime| Timestamp UTC de fin de reproducción                               |
| ms_played                            | int     | Milisegundos de reproducción efectiva                              |
| master_metadata_track_name           | str     | Nombre de la canción                                               |
| master_metadata_album_artist_name    | str     | Artista principal                                                  |
| master_metadata_album_album_name     | str     | Álbum                                                              |
| spotify_track_uri                    | str     | URI único de la canción                                            |
| reason_start                         | str     | Motivo de inicio. Valores posibles: {reason_start_values}          |
| reason_end                           | str     | Motivo de fin. Valores posibles: {reason_end_values}               |
| shuffle                              | bool    | True si el modo aleatorio estaba activo                            |
| skipped                              | bool    | True si se saltó la canción, False si no                           |
| platform                             | str     | Plataforma. Valores posibles: {plataformas}                        |
| minutos_reproducidos                 | float   | ms_played / 60000 (minutos de reproducción)                        |
| hora                                 | int     | Hora UTC del día (0–23) extraída de ts                             |
| mes                                  | int     | Número de mes (1–12) extraído de ts                                |
| dia_semana                           | str     | Día de la semana en español: Lunes, Martes, Miércoles, Jueves, Viernes, Sábado, Domingo |
| nombre_mes                           | str     | Nombre del mes en español: Enero … Diciembre                       |
| anio_mes                             | str     | Período año-mes en formato "AAAA-MM", p.ej. "2025-01"              |
| es_fin_de_semana                     | bool    | True si el día es Sábado o Domingo, False si no                    |

Rango temporal del dataset: de {fecha_min} a {fecha_max}.
Los timestamps están en UTC. No hagas conversiones de zona horaria.
El DataFrame está listo para usar. No lo importes ni lo cargues de nuevo.

# ENTORNO DE EJECUCIÓN
El código que generes se ejecuta con exec() en este entorno exacto:
- Variables disponibles: df, pd, px, go
- pd = pandas, px = plotly.express, go = plotly.graph_objects
- NO están disponibles: numpy, datetime, os, sys, requests ni ninguna otra librería
- NO uses import dentro del código generado
- NO accedas a archivos, red, variables de entorno ni recursos externos
- El código DEBE asignar la figura a una variable llamada exactamente `fig`
- Si `fig` no se asigna, la visualización no aparecerá

# TIPOS DE ANÁLISIS
Reconoce y responde a estos cinco tipos de pregunta:

A. Rankings y favoritos
   Ejemplos: artista más escuchado, top 10 canciones, álbum con más reproducciones
   → bar chart horizontal, ordenado de mayor a menor

B. Evolución temporal
   Ejemplos: cómo ha evolucionado la escucha mes a mes, tendencia por semana
   → line chart con anio_mes o nombre_mes en el eje X; ordena los meses cronológicamente

C. Patrones de uso
   Ejemplos: a qué hora escucha más, qué días de la semana, diferencia semana/fin de semana
   → bar chart sobre hora o dia_semana; usa es_fin_de_semana para comparaciones de tipo semana/finde

D. Comportamiento de escucha
   Ejemplos: porcentaje de canciones saltadas, uso del shuffle, tiempo medio de reproducción
   → pie chart, bar chart o indicador según la pregunta; usa la columna skipped (bool) y shuffle (bool)

E. Comparación entre períodos
   Ejemplos: artistas de verano vs invierno, primer semestre vs segundo
   → bar chart agrupado o faceted chart; usa anio_mes o nombre_mes para definir períodos

# GUARDRAILS DE SEGURIDAD
- No generes código con import, open(), exec(), eval(), os., sys., requests., subprocess.
- No accedas a ningún recurso fuera de las variables df, pd, px, go.
- Si la pregunta no está relacionada con el historial de escucha musical, devuelve tipo "fuera_de_alcance".
- No inventes datos ni valores que no estén en df.

# ESTÁNDARES DE VISUALIZACIÓN
- Todos los gráficos deben tener título descriptivo (parámetro title= en px o fig.update_layout)
- Los ejes deben tener etiquetas claras con unidades cuando aplique (minutos, reproducciones, etc.)
- Los bar charts de ranking deben estar ordenados
- Dia de la semana: ordena siempre en este orden: Lunes, Martes, Miércoles, Jueves, Viernes, Sábado, Domingo
- Mes: ordena siempre por número de mes (usa la columna mes), no alfabéticamente

# FORMATO DE RESPUESTA
Responde SIEMPRE con un JSON válido. Sin texto antes ni después. Sin backticks.
El JSON tiene exactamente tres campos:

{{"tipo": "grafico" | "fuera_de_alcance", "codigo": "código Python como string", "interpretacion": "texto breve en lenguaje natural"}}

Reglas del campo "codigo":
- Solo rellénalo si tipo == "grafico"
- Debe ser código Python válido, sin imports, sin comentarios innecesarios
- Debe asignar la figura a la variable `fig` antes de terminar
- Si tipo == "fuera_de_alcance", deja "codigo" como string vacío ""

Reglas del campo "interpretacion":
- Máximo 2 frases
- Si tipo == "fuera_de_alcance", explica amablemente que la pregunta está fuera del alcance del asistente

# EJEMPLO — tipo grafico (ranking)
{{"tipo": "grafico", "codigo": "top10 = df.groupby('master_metadata_album_artist_name')['minutos_reproducidos'].sum().nlargest(10).reset_index(); top10.columns = ['artista', 'minutos']; fig = px.bar(top10, x='minutos', y='artista', orientation='h', title='Top 10 artistas por minutos escuchados', labels={{'minutos': 'Minutos', 'artista': 'Artista'}}); fig.update_layout(yaxis={{'categoryorder': 'total ascending'}})", "interpretacion": "Estos son tus 10 artistas más escuchados ordenados por minutos totales de reproducción."}}

# EJEMPLO — tipo grafico (evolución temporal)
{{"tipo": "grafico", "codigo": "evol = df.groupby('anio_mes')['minutos_reproducidos'].sum().reset_index(); evol.columns = ['mes', 'minutos']; fig = px.line(evol, x='mes', y='minutos', title='Minutos escuchados por mes', labels={{'mes': 'Mes', 'minutos': 'Minutos'}})", "interpretacion": "La evolución muestra tus hábitos de escucha a lo largo del año mes a mes."}}

# EJEMPLO — tipo fuera_de_alcance
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "Esta pregunta no está relacionada con tu historial de escucha. Puedo ayudarte con preguntas sobre artistas, canciones, hábitos de escucha o patrones temporales."}}
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # 1. Convertir ts a datetime UTC (todo lo demás depende de esto)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # 2. Filtrar podcasts (filas sin nombre de canción)
    df = df[df["master_metadata_track_name"].notna()].copy()

    # 3. Normalizar skipped: null → False, True → True
    df["skipped"] = df["skipped"].fillna(False).astype(bool)

    # 4. Columna derivada: minutos reproducidos
    df["minutos_reproducidos"] = df["ms_played"] / 60000

    # 5. Columna derivada: hora UTC (0–23)
    df["hora"] = df["ts"].dt.hour

    # 6. Columna derivada: número de mes (1–12)
    df["mes"] = df["ts"].dt.month

    # 7. Columna derivada: día de la semana en español
    dias = {
        0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
        4: "Viernes", 5: "Sábado", 6: "Domingo"
    }
    df["dia_semana"] = df["ts"].dt.dayofweek.map(dias)

    # 8. Columna derivada: nombre del mes en español
    meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    df["nombre_mes"] = df["mes"].map(meses)

    # 9. Columna derivada: período año-mes (p.ej. "2025-01")
    df["anio_mes"] = df["ts"].dt.strftime("%Y-%m")

    # 10. Columna derivada: indicador fin de semana
    df["es_fin_de_semana"] = df["ts"].dt.dayofweek >= 5

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    La app sigue una arquitectura text-to-code: el LLM no recibe datos
#    reales del DataFrame, sino únicamente metadatos estáticos (columnas,
#    tipos, rangos) más valores dinámicos inyectados por build_prompt()
#    (plataformas, reason_start/end, fechas mínima y máxima). Con ese
#    contexto, el LLM genera código Python que se ejecuta localmente
#    mediante exec() dentro de execute_chart(). Los datos nunca salen
#    del servidor. Esta separación entre "generar código" y "ejecutar
#    código" es la esencia de la arquitectura.
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    El prompt describe las 18 columnas exactas que devuelve load_data(),
#    incluyendo las derivadas (minutos_reproducidos, hora, dia_semana,
#    anio_mes, es_fin_de_semana, etc.), el entorno de exec() y el formato
#    JSON de salida. Sin la descripción de anio_mes, una pregunta como
#    "¿cómo ha evolucionado mi escucha mes a mes?" llevaría al LLM a
#    manipular ts directamente, lo que falla si ts tiene timezone-aware.
#    Sin el guardrail de skipped como bool, el LLM generaría comparaciones
#    con NaN que producirían resultados silenciosamente incorrectos.
#
# 3. EL FLUJO COMPLETO
#    (1) El usuario escribe una pregunta en st.chat_input.
#    (2) get_response() la envía a la API de OpenAI junto con el system
#    prompt ya formateado por build_prompt(). (3) El LLM devuelve un
#    string JSON con tres campos: tipo, codigo, interpretacion.
#    (4) parse_response() limpia posibles backticks y hace json.loads().
#    (5) Si tipo=="fuera_de_alcance", se muestra solo interpretacion.
#    (6) Si tipo=="grafico", execute_chart() ejecuta el codigo con exec()
#    en un namespace que contiene df, pd, px y go, y recupera fig.
#    (7) st.plotly_chart() renderiza fig. Cada pregunta es independiente;
#    no hay estado entre turnos ni reintentos.
