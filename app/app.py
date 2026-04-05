import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, MATCH
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import traceback
import re
import logging
import time
import uuid

CONFIG = {
'MAX_ROWS_FOR_CLIENT': 50000,
'MAX_FILE_SIZE_MB': 50,
'SAMPLE_RANDOM_STATE': 42,
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processors.data_analyzer import DataAnalyzer
from processors.nlp_processor import NLPProcessor
from processors.viz_generator import VizGenerator

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .fade-in {
                animation: fadeIn 0.4s ease-in-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .card-hover {
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .card-hover:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
            }
            .btn-pulse {
                transition: all 0.2s ease;
            }
            .btn-pulse:hover {
                transform: scale(1.02);
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }
            .custom-spinner div {
                border-color: #2563eb !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
server = app.server
app.title = "CSV Analyzer - Análisis automático de datos"

def get_dataframe(df_data, df_id=None, use_server_if_large=True):
    if df_data and isinstance(df_data, list) and len(df_data) > 0:
        df_client = pd.DataFrame(df_data)
        if use_server_if_large and df_id and hasattr(app, '_server_dataframes') and df_id in app._server_dataframes:
            df_server = app._server_dataframes[df_id]
            meta_key = f"{df_id}_meta"
            if meta_key in app._server_dataframes and app._server_dataframes[meta_key].get('sampled'):
                logger.debug(f"Usando DataFrame completo del servidor ({len(df_server)} filas)")
                return df_server
        return df_client
    if df_id and hasattr(app, '_server_dataframes') and df_id in app._server_dataframes:
        return app._server_dataframes[df_id]
    logger.warning("get_dataframe: No se pudo recuperar ningún DataFrame")
    return pd.DataFrame()

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📊 CSV Analyzer", className="text-center mt-4 mb-4"),
            html.P("Subí tu archivo CSV y obtené análisis automático",
                   className="text-center text-muted mb-5"),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.Div("📂", style={"fontSize": "64px", "marginBottom": "8px", "lineHeight": "1"}),
                    html.P("Arrastrá o hacé clic para seleccionar un archivo CSV",
                           style={"margin": "0", "fontSize": "16px", "color": "#555", "fontWeight": "500", "textAlign": "center"})
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'height': '160px',
                    'width': '100%'
                }),
                style={
                    'width': '100%',
                    'border': '2px dashed #ced4da',
                    'borderRadius': '15px',
                    'backgroundColor': '#f8f9fa',
                    'cursor': 'pointer',
                    'textAlign': 'center',
                    'padding': '10px',
                    'boxSizing': 'border-box'
                },
                multiple=False,
                accept='.csv'
            )
        ], width=12, className="px-3")
    ], className="mt-2"),
    html.Div(id='output-data-upload', className="mt-4"),
    dcc.Store(id='report-store', data=[]),
    dcc.Store(id='accumulated-charts-ids', data=[]),
], fluid=True)

def detect_column_type(df, col):
    if pd.api.types.is_numeric_dtype(df[col]):
        return 'numeric'
    else:
        unique_values = df[col].dropna().unique()
        ordinal_patterns = [
            'muy de acuerdo', 'de acuerdo', 'en desacuerdo', 'muy en desacuerdo',
            'siempre', 'casi siempre', 'a veces', 'nunca',
            'excelente', 'bueno', 'regular', 'malo',
            'alto', 'medio', 'bajo'
        ]
        lower_values = [str(v).lower() for v in unique_values]
        is_ordinal = any(any(pattern in val for pattern in ordinal_patterns) for val in lower_values)
        return 'ordinal' if is_ordinal else 'categorical'

def create_adaptive_chart(df, chart_type, col_a, col_b, col_c=None, color_palette='Viridis'):
    type_a = detect_column_type(df, col_a)
    type_b = detect_column_type(df, col_b) if col_b else None
    type_c = detect_column_type(df, col_c) if col_c else None

    def get_color_param(use_col_c, col_c_type, palette):
        if not use_col_c or col_c_type is None:
            return {}
        continuous_scales = ['Viridis', 'Plasma', 'Cividis', 'Magma', 'Inferno']
        discrete_sequences = {
            'Viridis': px.colors.sequential.Viridis,
            'Plasma': px.colors.sequential.Plasma,
            'Cividis': px.colors.sequential.Cividis,
            'Magma': px.colors.sequential.Magma,
            'Inferno': px.colors.sequential.Inferno,
        }
        if col_c_type == 'numeric':
            return {'color_continuous_scale': palette if palette in continuous_scales else 'Viridis'}
        else:
            seq = discrete_sequences.get(palette, px.colors.qualitative.Plotly)
            return {'color_discrete_sequence': seq}
    
    if chart_type == 'heatmap':
        if type_a == 'numeric' and type_b == 'numeric':
            clean_df = df[[col_a, col_b]].dropna()
            if len(clean_df) < 2:
                return go.Figure().add_annotation(text="Datos insuficientes")
            corr = clean_df[col_a].corr(clean_df[col_b])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=clean_df[col_a], y=clean_df[col_b], mode='markers', marker=dict(size=8, opacity=0.6)))
            z = np.polyfit(clean_df[col_a], clean_df[col_b], 1)
            fig.add_trace(go.Scatter(x=np.linspace(clean_df[col_a].min(), clean_df[col_a].max(), 100), 
                                    y=np.poly1d(z)(np.linspace(clean_df[col_a].min(), clean_df[col_a].max(), 100)), 
                                    mode='lines', line=dict(color='red', width=2), name=f'Tendencia (r={corr:.3f})'))
            fig.update_layout(title=f"Correlación: {col_a} vs {col_b}<br><sup>r = {corr:.3f}</sup>", xaxis_title=col_a, yaxis_title=col_b)
            return fig
        elif (type_a == 'numeric' and type_b in ['categorical', 'ordinal']) or (type_b == 'numeric' and type_a in ['categorical', 'ordinal']):
            num_col, cat_col = (col_a, col_b) if type_a == 'numeric' else (col_b, col_a)
            fig = px.box(df, x=cat_col, y=num_col, color=cat_col, 
                        title=f"Distribución de {num_col} por {cat_col}",
                        color_discrete_sequence=px.colors.qualitative.Plotly)
            fig.update_traces(boxmean=True)
            return fig
        else:
            contingency = pd.crosstab(df[col_a], df[col_b])
            fig = px.imshow(contingency, text_auto=True, aspect="auto", 
                        title=f"Frecuencias: {col_a} vs {col_b}", 
                        color_continuous_scale=color_palette)
            return fig

    elif chart_type == 'scatter':
        if col_c:
            color_col = col_c
            color_type = type_c
        elif type_b in ['categorical', 'ordinal']:
            color_col = col_b
            color_type = type_b
        else:
            color_col = None
            color_type = None
        color_params = get_color_param(color_col is not None, color_type, color_palette) if color_col else {}
        fig = px.scatter(df, x=col_a, y=col_b, color=color_col, title=f"{col_a} vs {col_b}", **color_params)
        fig.update_traces(marker_size=8)
        return fig

    elif chart_type == 'line':
        if col_c:
            color_col = col_c
            color_type = type_c
        elif type_b in ['categorical', 'ordinal']:
            color_col = col_b
            color_type = type_b
        else:
            color_col = None
            color_type = None
        color_params = get_color_param(color_col is not None, color_type, color_palette) if color_col else {}
        fig = px.line(df, x=col_a, y=col_b, color=color_col, title=f"Tendencia: {col_a} vs {col_b}", **color_params)
        fig.update_traces(line_width=2)
        return fig

    elif chart_type == 'histogram':
        if type_a == 'numeric':
            return px.histogram(df, x=col_a, title=f"Distribución de {col_a}", nbins=30, color_discrete_sequence=['steelblue'])
        else:
            freq = df[col_a].value_counts().reset_index()
            freq.columns = [col_a, 'Frecuencia']
            return px.bar(freq, x=col_a, y='Frecuencia', title=f"Frecuencias de {col_a}", color_discrete_sequence=['steelblue'])

    elif chart_type == 'box':
        if col_b:
            fig = px.box(df, x=col_b, y=col_a, color=col_b, title=f"Distribución de {col_a} por {col_b}",
                        color_discrete_sequence=px.colors.qualitative.Plotly)
            fig.update_traces(boxmean=True)
        else:
            fig = px.box(df, y=col_a, title=f"Distribución de {col_a}")
            fig.update_traces(boxmean=True)
        return fig

    elif chart_type == 'bar':
        freq = df[col_a].value_counts().reset_index()
        freq.columns = [col_a, 'Frecuencia']
        return px.bar(freq, x=col_a, y='Frecuencia', title=f"Frecuencias de {col_a}", color_discrete_sequence=['steelblue'])

    return go.Figure()

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
                logger.info(f"Archivo procesado: {filename}, {len(df)} filas, {len(df.columns)} columnas")
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("No se pudo decodificar el archivo. Verificá el encoding del CSV.")

        analyzer = DataAnalyzer(df)
        summary = analyzer.get_summary()
        stats = analyzer.get_statistics()
        viz = VizGenerator(df)
        text_columns = analyzer.get_text_columns()
        for col in df.columns:
            if col.startswith('año_'):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        children = [
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H4(f"✅ {filename}", className="card-title mb-2 fade-in"),
                            html.Div([
                                html.P([html.Strong("📊 Filas: "), f"{summary['rows']:,}"], className="mb-1 fade-in"),
                                html.P([html.Strong("🔢 Columnas: "), summary['columns']], className="mb-1 fade-in"),
                                html.P([html.Strong("🗂️ Numéricas: "), summary['numeric_cols']], className="mb-0 fade-in"),
                            ], className="ps-2")
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.P([html.Strong("📝 Texto: "), summary['text_cols']], className="mb-1 fade-in"),
                                html.P([html.Strong("🏷️ Categóricas: "), summary['categorical_cols']], className="mb-1 fade-in"),
                                html.P([html.Strong("⚠️ Nulos: "), f"{summary['null_counts']} total"], className="mb-0 fade-in"),
                            ], className="ps-2")
                        ], width=6)
                    ], className="fade-in")
                ])
            ], className="mb-4 border-primary shadow-sm fade-in card-hover"),
            
            html.H4("📈 Estadísticas por columna numérica", className="fade-in"),
            html.P("Seleccioná una columna para ver sus estadísticas descriptivas", className="text-muted mb-3 fade-in"),
        ]
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 0:
            children.extend([
                dbc.Row([
                    dbc.Col([
                        html.Label("📁 Columna numérica:"),
                        dcc.Dropdown(
                            id='stats-col-select',
                            options=[{'label': col, 'value': col} for col in numeric_cols],
                            value=numeric_cols[0],
                            clearable=False
                        )
                    ], width=8),
                    dbc.Col([
                        html.Label(" "),
                        html.Button("🔍 Ver estadísticas", id="generate-stats", 
                                   n_clicks=0, className="btn btn-primary w-100 btn-pulse")
                    ], width=4)
                ], className="mb-3 fade-in"),
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-stats",
                            type="circle",
                            children=html.Div(id="stats-output", className="mt-3", style={'minHeight': '200px'}),
                            className="custom-spinner"
                        )
                    ], width=12)
                ]),
            ])
        
        if len(numeric_cols) > 0:
            children.extend([
                html.Hr(className="fade-in"),
                html.H4("📊 Gráficos de columnas numéricas", className="fade-in"),
                html.P("Seleccioná una columna y el tipo de gráfico para visualizar", className="text-muted mb-3 fade-in"),
                dbc.Row([
                    dbc.Col([
                        html.Label("📁 Columna numérica:"),
                        dcc.Dropdown(id='numeric-col-select', options=[{'label': col, 'value': col} for col in numeric_cols], value=numeric_cols[0], clearable=False)
                    ], width=4),
                    dbc.Col([
                        html.Label("📊 Tipo de gráfico:"),
                        dcc.Dropdown(id='numeric-chart-type', options=[
                            {'label': '📊 Histograma', 'value': 'histogram'},
                            {'label': '📈 Densidad (KDE)', 'value': 'kde'},
                            {'label': '📉 Línea (por índice)', 'value': 'line'}
                        ], value='histogram', clearable=False)
                    ], width=4),
                    dbc.Col([
                        html.Label("🎨 Paleta:"),
                        dcc.Dropdown(id='numeric-palette', options=[
                            {'label': '🌿 Viridis', 'value': 'Viridis'},
                            {'label': '🔥 Plasma', 'value': 'Plasma'},
                            {'label': '💚 Cividis', 'value': 'Cividis'},
                        ], value='Viridis', clearable=False)
                    ], width=4),
                ], className="mb-3 fade-in"),
                dbc.Row([
                    dbc.Col([html.Label("🔍 Filtrar por columna:")], width=6),
                    dbc.Col([html.Label("🎯 Valor del filtro:")], width=6),
                ], className="mb-1 fade-in"),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(id='numeric-filter-col', options=[{'label': col, 'value': col} for col in df.columns], placeholder="Ninguna (todos los datos)", clearable=True)
                    ], width=6),
                    dbc.Col([
                        dcc.Dropdown(id='numeric-filter-val', options=[], placeholder="Seleccioná columna filtro primero", clearable=True)
                    ], width=6),
                ], className="mb-3 fade-in"),
                dbc.Row([
                    dbc.Col([
                        html.Button("🎨 Generar gráfico", id="generate-numeric-chart", n_clicks=0, className="btn btn-primary btn-pulse")
                    ], width=12)
                ], className="mb-3 fade-in"),
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(id="loading-numeric-chart", type="circle", 
                                   children=html.Div(id="numeric-chart-output", className="mt-3", style={'minHeight': '300px'}),
                                   className="custom-spinner")
                    ], width=12)
                ]),
            ])
        
        categorical_cols = analyzer.get_categorical_columns()
        if len(categorical_cols) > 0:
            children.extend([
                html.Hr(className="fade-in"),
                html.H4("🏷️ Gráficos de columnas categóricas (a pedido)", className="fade-in"),
                html.P("Seleccioná una columna categórica para visualizar su distribución", className="text-muted mb-3 fade-in"),
                dbc.Row([
                    dbc.Col([
                        html.Label("📁 Columna a graficar:"),
                        dcc.Dropdown(id='categorical-col-select', options=[{'label': col, 'value': col} for col in categorical_cols], value=categorical_cols[0], clearable=False)
                    ], width=6),
                    dbc.Col([
                        html.Label("📊 Tipo de gráfico:"),
                        dcc.Dropdown(id='categorical-chart-type', options=[
                            {'label': '📊 Barras (frecuencia)', 'value': 'bar'},
                            {'label': '🥧 Torta (pie)', 'value': 'pie'},
                            {'label': '📦 Treemap', 'value': 'treemap'},
                        ], value='bar', clearable=False)
                    ], width=6),
                ], className="mb-3 fade-in"),
                dbc.Row([
                    dbc.Col([html.Label("🔍 Filtrar por columna:")], width=6),
                    dbc.Col([html.Label("🎯 Valor del filtro:")], width=6),
                ], className="mb-1 fade-in"),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(id='categorical-filter-col', options=[{'label': col, 'value': col} for col in df.columns], placeholder="Ninguna (todos los datos)", clearable=True)
                    ], width=6),
                    dbc.Col([
                        dcc.Dropdown(id='categorical-filter-val', options=[], placeholder="Seleccioná columna filtro primero", clearable=True)
                    ], width=6),
                ], className="mb-3 fade-in"),
                dbc.Row([
                    dbc.Col([
                        html.Button("🎨 Generar gráfico", id="generate-categorical-chart", n_clicks=0, className="btn btn-primary btn-pulse")
                    ], width=12)
                ], className="mb-3 fade-in"),
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(id="loading-categorical-chart", type="circle", 
                                   children=html.Div(id="categorical-chart-output", className="mt-3", style={'minHeight': '300px'}),
                                   className="custom-spinner")
                    ], width=12)
                ]),
            ])

        children.extend([
            html.Hr(className="fade-in"),
            html.H4("📝 Análisis de texto", className="fade-in"),
        ])

        if text_columns:
            def is_good_text_column(df, col):
                sample = df[col].dropna().astype(str).head(20)
                word_counts = sample.apply(lambda x: len(re.findall(r'\b[a-zA-Z]{3,}\b', x)))
                return word_counts.mean() >= 2

            valid_text_cols = [col for col in text_columns if is_good_text_column(df, col)]
            dropdown_options = [
                {'label': f"{col} ✅" if col in valid_text_cols else f"{col} ⚠️", 'value': col} 
                for col in text_columns
            ]

            children.extend([
                html.P("Seleccioná la columna que contiene texto para analizar:", className="fade-in"),
                html.Small("✅ Columnas recomendadas para análisis de sentimiento", className="text-muted d-block mb-2 fade-in"),
                dcc.Dropdown(
                    id='text-column-selector',
                    options=dropdown_options,
                    value=valid_text_cols[0] if valid_text_cols else text_columns[0],
                    clearable=False,
                    style={'marginBottom': '20px'}
                ),
                html.Button("🔍 Analizar sentimiento", id="analyze-sentiment-btn", n_clicks=0,
                            className="btn btn-secondary mt-2 mb-3 btn-pulse"),
            ])
        else:
            children.append(html.P("⚠️ No se detectaron columnas de texto en este archivo.", className="text-muted fade-in"))

        children.append(html.Div(id='nlp-results-content', className="mt-3 mb-4"))

        children.append(html.Hr(className="fade-in"))
        children.append(html.H3("🎨 Visualización interactiva", className="mt-4 mb-3 fade-in"))
        children.append(html.P("Seleccioná las columnas y el tipo de gráfico - se adapta automáticamente al tipo de datos", className="fade-in"))

        children.append(
            dbc.Row([
                dbc.Col([
                    html.Label("📊 Tipo de gráfico:"),
                    dcc.Dropdown(id='chart-type', options=[
                        {'label': '🔥 Heatmap', 'value': 'heatmap'},
                        {'label': '🔵 Scatter', 'value': 'scatter'},
                        {'label': '📉 Línea', 'value': 'line'},
                        {'label': '📊 Histograma', 'value': 'histogram'},
                        {'label': '📦 Box plot', 'value': 'box'},
                        {'label': '📊 Barras', 'value': 'bar'}
                    ], value='heatmap', clearable=False)
                ], width=3),
                dbc.Col([
                    html.Label("📁 Columna A (X):"),
                    dcc.Dropdown(id='col-a', options=[{'label': c, 'value': c} for c in df.columns], placeholder="Eje X")
                ], width=3),
                dbc.Col([
                    html.Label("📁 Columna B (Y):"),
                    dcc.Dropdown(id='col-b', options=[{'label': c, 'value': c} for c in df.columns], placeholder="Eje Y"),
                ], width=3, id='col-b-col'),
                dbc.Col([
                    html.Label("🎨 Columna C (Color):"),
                    dcc.Dropdown(id='col-c', options=[], placeholder="Seleccioná A y B primero", clearable=True),
                ], width=3, id='col-c-col'),
            ], className="fade-in")
        )
        children.append(
            dbc.Row([
                dbc.Col([
                    html.Label("🌈 Paleta de colores:"),
                    dcc.Dropdown(id='color-palette', options=[
                        {'label': '🌿 Viridis', 'value': 'Viridis'},
                        {'label': '🔥 Plasma', 'value': 'Plasma'},
                        {'label': '💚 Cividis', 'value': 'Cividis'},
                        {'label': '🌋 Magma', 'value': 'Magma'},
                        {'label': '🏔️ Inferno', 'value': 'Inferno'}
                    ], value='Viridis', clearable=False)
                ], width=4),
                dbc.Col([
                    html.Button("🎨 Generar gráfico", id="generate-chart", n_clicks=0, className="btn btn-primary mt-3 btn-pulse"),
                ], width=8, className="d-flex align-items-end")
            ], className="fade-in")
        )
        children.append(
            dbc.Row([
                dbc.Col([
                    dcc.Loading(id="loading-chart", type="default", children=html.Div(id="generated-chart", className="mt-4"), className="custom-spinner")
                ], width=12)
            ])
        )
        
        children.append(html.Div(id='accumulated-charts-container', className="mt-4"))

        numeric_cols_list = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols_list = df.select_dtypes(include=['object']).columns.tolist()
        all_columns_list = df.columns.tolist()

        total_rows = len(df)
        if total_rows > CONFIG['MAX_ROWS_FOR_CLIENT']:
            logger.info(f"Dataset grande ({total_rows} filas). Limitando a {CONFIG['MAX_ROWS_FOR_CLIENT']} filas.")
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                stratify_col = cat_cols[0]
                if df[stratify_col].nunique() < 100 and df[stratify_col].notna().mean() > 0.9:
                    n_per_group = max(1, CONFIG['MAX_ROWS_FOR_CLIENT'] // df[stratify_col].nunique())
                    df_client = df.groupby(stratify_col, group_keys=False).apply(
                        lambda x: x.sample(n=min(len(x), n_per_group), random_state=CONFIG['SAMPLE_RANDOM_STATE'])
                    ).reset_index(drop=True)
                else:
                    df_client = df.sample(n=CONFIG['MAX_ROWS_FOR_CLIENT'], random_state=CONFIG['SAMPLE_RANDOM_STATE']).reset_index(drop=True)
            else:
                df_client = df.sample(n=CONFIG['MAX_ROWS_FOR_CLIENT'], random_state=CONFIG['SAMPLE_RANDOM_STATE']).reset_index(drop=True)
            
            sample_info = html.Div([
                dbc.Alert([
                    html.I(className="bi bi-info-circle-fill me-2"),
                    f"📊 Mostrando muestra de {len(df_client):,} de {total_rows:,} filas para mejor rendimiento.",
                ], color="info", className="mb-3 fade-in")
            ])
            children.insert(0, sample_info)
            sampled = True
        else:
            df_client = df
            logger.info(f"Dataset pequeño ({total_rows} filas). Usando completo.")
            sampled = False

        df_id = str(uuid.uuid4())
        if not hasattr(app, '_server_dataframes'):
            app._server_dataframes = {}
        app._server_dataframes[df_id] = df
        app._server_dataframes[f"{df_id}_meta"] = {'rows': total_rows, 'sampled': sampled, 'timestamp': time.time()}

        children.append(dcc.Store(id='numeric-columns', data=numeric_cols_list))
        children.append(dcc.Store(id='categorical-columns', data=categorical_cols_list))
        children.append(dcc.Store(id='df-store', data=df_client.to_dict('records')))
        children.append(dcc.Store(id='text-columns-store', data=text_columns))
        children.append(dcc.Store(id='all-columns-store', data=all_columns_list))
        children.append(dcc.Store(id='df-server-id', data=df_id if sampled else None))
        
        return html.Div(children)

    except Exception as e:
        return html.Div([
            html.H4("❌ Error al procesar el archivo", className="text-danger fade-in"),
            html.P(str(e)),
            html.Details(html.Pre(traceback.format_exc()))
        ])

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=False
)
def update_output(contents, filename):
    logger.info(f"📥 update_output DISPARADO | filename={filename}")
    if not contents:
        if filename and filename.endswith('.csv'):
            return html.Div("🔄 Archivo seleccionado. Procesando...", className="text-info")
        return html.Div()
    try:
        if ',' not in str(contents):
            return html.Div("❌ Formato de archivo no reconocido", className="text-danger")
        content_type, content_string = contents.split(',')
        if len(content_string) > 50 * 1024 * 1024:
            return html.Div([dbc.Alert("❌ Archivo demasiado grande (máx. 50MB)", color="danger", className="mt-3")])
        result = parse_contents(contents, filename)
        return result
    except Exception as e:
        logger.error(f"❌ ERROR en update_output: {e}", exc_info=True)
        return html.Div([
            dbc.Alert([
                html.H4("❌ Error al procesar", className="alert-heading"),
                html.P(f"{str(e)}"),
                html.Small("Revisá terminal para detalles", className="text-muted")
            ], color="danger", className="mt-3")
        ])

@app.callback(
    Output('nlp-results-content', 'children'),
    Input('analyze-sentiment-btn', 'n_clicks'),
    [State('text-column-selector', 'value'),
     State('df-store', 'data'),
     State('df-server-id','data')],
    prevent_initial_call=True
)
def update_nlp_analysis(n_clicks, selected_column, df_data, df_server_id):
    if not df_data or not selected_column:
        return html.Div("⚠️ No hay datos o columna seleccionada", className="text-muted")
    try:
        df = get_dataframe(df_data, df_id=df_server_id, use_server_if_large=True)
        if df.empty:
            return html.Div("⚠️ Los datos están vacíos.", className="text-warning")
        if selected_column not in df.columns:
            return html.Div(f"⚠️ La columna '{selected_column}' no existe.", className="text-warning")
        if df[selected_column].dtype == 'object':
            non_empty = df[selected_column].dropna().astype(str)
            has_valid_text = non_empty.apply(lambda x: len(str(x).strip()) > 3).any()
            if not has_valid_text:
                return html.Div(f"⚠️ La columna '{selected_column}' no contiene texto analizable.", className="text-warning")
        else:
            return html.Div(f"⚠️ La columna '{selected_column}' no es de tipo texto.", className="text-warning")
        
        non_empty = df[selected_column].dropna()
        logger.info(f"Iniciando análisis NLP: columna='{selected_column}', {len(non_empty)} textos")

        nlp = NLPProcessor(df, selected_column)
        nlp_results = nlp.analyze()
        viz = VizGenerator(df)
        sentiment_gauge = viz.create_sentiment_gauge(nlp_results['avg_sentiment'])
        
        sentiment_score = nlp_results['avg_sentiment']
        if sentiment_score > 0.3:
            sentiment_icon, sentiment_color = "😊", "green"
        elif sentiment_score < -0.3:
            sentiment_icon, sentiment_color = "😞", "red"
        else:
            sentiment_icon, sentiment_color = "😐", "orange"
        
        wordcloud_component = None
        wordcloud_data = nlp_results.get('wordcloud')
        if wordcloud_data and len(wordcloud_data) >= 1:
            try:
                if isinstance(wordcloud_data, dict):
                    wordcloud_dict = dict(wordcloud_data)
                else:
                    wordcloud_dict = {}
                if wordcloud_dict:
                    img_base64 = viz.create_wordcloud_plot(wordcloud_dict)
                    if img_base64 and len(img_base64) > 1000:
                        img_src = f'data:image/png;base64,{img_base64}'
                        wordcloud_component = html.Div([
                            html.Img(src=img_src, style={'width': '100%', 'maxHeight': '400px', 'objectFit': 'contain'}),
                            html.Small(f"✅ Imagen cargada: {len(img_base64)} caracteres", className="text-success d-block mt-1")
                        ])
                    else:
                        wordcloud_component = html.P("⚠️ La imagen base64 está vacía o es muy pequeña", className="text-warning")
                else:
                    wordcloud_component = html.P("⚠️ No hay palabras válidas para mostrar", className="text-muted")
            except Exception as e:
                logger.info(f"Error renderizando wordcloud: {e}")
                wordcloud_component = html.P(f"⚠️ Error: {str(e)}", className="text-danger")
        else:
            wordcloud_component = html.P("⚠️ El análisis no generó datos para la nube", className="text-muted")
        
        results = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H4(f"📝 Análisis de: {selected_column}", className="mb-1"),
                    html.P(f"📊 {len(non_empty)} respuestas analizadas", className="text-muted mb-0")
                ], className="d-flex justify-content-between align-items-center"),
                html.Hr(className="my-3"),
                dbc.Row([
                    dbc.Col([
                        html.H6("🎯 Sentimiento promedio", className="mb-2"),
                        dcc.Graph(figure=sentiment_gauge, config={'displayModeBar': False})
                    ], width=5),
                    dbc.Col([
                        html.H6("📊 Distribución", className="mb-2"),
                        html.Div([
                            html.P(f"{sentiment_icon} General: ", style={"display": "inline", "fontWeight": "bold"}),
                            html.Span(f"{sentiment_score:.3f}", style={"color": sentiment_color, "fontWeight": "bold", "fontSize": "1.2rem"}),
                        ], className="mb-2"),
                        html.Div([
                            dbc.Progress([
                                dbc.Progress(value=nlp_results['positive_percent'], className="bg-success", striped=True),
                                dbc.Progress(value=nlp_results['neutral_percent'], className="bg-warning", striped=True),
                                dbc.Progress(value=nlp_results['negative_percent'], className="bg-danger", striped=True),
                            ], style={'height': '20px'}, className="mb-2"),
                            html.Small([
                                html.Span(f"😊 {nlp_results['positive_percent']:.1f}% ", className="text-success"),
                                html.Span(f"😐 {nlp_results['neutral_percent']:.1f}% ", className="text-warning"),
                                html.Span(f"😞 {nlp_results['negative_percent']:.1f}%", className="text-danger"),
                            ])
                        ]),
                        html.Hr(className="my-2"),
                        html.P(f"📖 Vocabulario: {nlp_results['vocab_size']} palabras únicas", className="mb-1"),
                        html.P("📝 Top 5 palabras:", className="mb-1"),
                        html.Ul([
                            html.Li(f"{w}: {c}", className="mb-0") for w, c in nlp_results['top_words'][:5]
                        ], style={"fontSize": "0.9rem"})
                    ], width=7)
                ]),
                html.Div([
                    html.Hr(className="my-3"),
                    html.H6("☁️ Nube de palabras más frecuentes", className="mb-2"),
                    wordcloud_component if wordcloud_component else html.P("⚠️ No hay palabras suficientes para generar la nube", className="text-muted")
                ], className="mt-3")
            ])
        ], className="shadow-sm")
        return results
    except Exception as e:
        return dbc.Alert([
            html.H4("❌ Error en el análisis", className="alert-heading"),
            html.P(str(e)),
            html.Details(html.Pre(traceback.format_exc(), style={"fontSize": "0.75rem"}))
        ], color="danger")

@app.callback(
    Output('generated-chart', 'children'),
    Input('generate-chart', 'n_clicks'),
    [State('chart-type', 'value'),
     State('col-a', 'value'),
     State('col-b', 'value'),
     State('col-c', 'value'),
     State('color-palette', 'value'),
     State('df-store', 'data')],
    prevent_initial_call=True
)
def generate_chart(n_clicks, chart_type, col_a, col_b, col_c, color_palette, df_data):
    if not df_data:
        return html.Div("⚠️ No hay datos cargados. Subí un archivo CSV primero.", className="text-warning")
    if not col_a:
        return html.Div("⚠️ Seleccioná al menos la Columna A (eje X).", className="text-warning")
    valid_chart_types = ['heatmap', 'scatter', 'line', 'histogram', 'box', 'bar']
    if chart_type not in valid_chart_types:
        chart_type = 'scatter'
    if chart_type in ['heatmap', 'scatter', 'line', 'box'] and not col_b:
        return html.Div(f"⚠️ El gráfico '{chart_type}' requiere seleccionar Columna A y Columna B.", className="text-warning")
    try:
        df = pd.DataFrame(df_data)
    except Exception as e:
        return html.Div("❌ Error interno al procesar los datos.", className="text-danger")
    valid_columns = set(df.columns)
    for col_name, col_value, required in [
        ('Columna A', col_a, True),
        ('Columna B', col_b, chart_type in ['heatmap', 'scatter', 'line']),
        ('Columna C', col_c, False)
    ]:
        if col_value and col_value not in valid_columns:
            return html.Div(f"⚠️ La columna '{col_value}' no existe en los datos cargados.", className="text-danger")
    valid_palettes = ['Viridis', 'Plasma', 'Cividis', 'Magma', 'Inferno']
    if color_palette not in valid_palettes:
        color_palette = 'Viridis'
    try:
        fig = create_adaptive_chart(df, chart_type, col_a, col_b, col_c, color_palette)
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div([html.H4("❌ Error", className="text-danger"), html.P(str(e))])

@app.callback(
    [Output('col-b', 'disabled'), Output('col-b', 'placeholder'), Output('col-b-col', 'style'),
     Output('col-c', 'disabled'), Output('col-c-col', 'style')],
    Input('chart-type', 'value')
)
def update_columns_visibility(chart_type):
    if chart_type in ['heatmap', 'scatter', 'line', 'box']:
        return False, "Seleccioná columna B", {'display': 'block'}, False, {'display': 'block'}
    else:
        return True, "No necesario", {'display': 'none'}, True, {'display': 'none'}

@app.callback(
    [Output('col-c', 'options'),
     Output('col-c', 'placeholder')],
    [Input('col-a', 'value'),
     Input('col-b', 'value'),
     Input('all-columns-store', 'data')],
    prevent_initial_call=True
)
def update_col_c_options(col_a, col_b, all_columns):
    if not all_columns:
        return [], "Seleccioná A y B primero"
    excluded = {col_a, col_b} - {None}
    available_cols = [col for col in all_columns if col not in excluded]
    if not available_cols:
        return [], "No hay columnas disponibles"
    options = [{'label': col, 'value': col} for col in available_cols]
    return options, f"Opcional ({len(available_cols)} columnas disponibles)"

@app.callback(
    Output('numeric-chart-output', 'children'),
    Input('generate-numeric-chart', 'n_clicks'),
    [State('numeric-col-select', 'value'),
     State('numeric-chart-type', 'value'),
     State('numeric-palette', 'value'),
     State('df-store', 'data'),
     State('numeric-filter-col', 'value'),
     State('numeric-filter-val', 'value')],
    prevent_initial_call=True
)
def generate_numeric_chart(n_clicks, col_name, chart_type, palette, df_data, filter_col, filter_val):
    if not df_data or not col_name:
        return html.Div("⚠️ No hay datos", className="text-warning")
    try:
        df = pd.DataFrame(df_data)
        if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]):
            return html.Div(f"⚠️ '{col_name}' no es válida", className="text-danger")
        filter_info = ""
        if filter_col and filter_val is not None and filter_col in df.columns:
            df = df[df[filter_col] == filter_val]
            if df.empty:
                return html.Div(f"⚠️ No hay datos para {filter_col}='{filter_val}'", className="text-warning")
            filter_info = f"<br><sup>(Filtrado: {filter_col} = '{filter_val}')</sup>"
        PALETTES = {'Viridis': px.colors.sequential.Viridis, 'Plasma': px.colors.sequential.Plasma, 'Cividis': px.colors.sequential.Cividis}
        color_scale = PALETTES.get(palette, px.colors.sequential.Viridis)
        color = color_scale[0] if isinstance(color_scale, list) else color_scale
        subtitle = f"<br><sup>({'Filtrado' if filter_info else 'Datos completos'})</sup>"
        if chart_type == 'histogram':
            fig = px.histogram(df, x=col_name, title=f"Distribución de {col_name}{subtitle}", nbins=30, color_discrete_sequence=[color])
        elif chart_type == 'kde':
            fig = px.histogram(df, x=col_name, title=f"Densidad de {col_name}{subtitle}", histnorm='probability density', nbins=50, color_discrete_sequence=[color])
            fig.update_traces(marker_line_width=0)
        elif chart_type == 'line':
            df_temp = df[col_name].reset_index()
            fig = px.line(df_temp, x='index', y=col_name, title=f"{col_name} por posición{subtitle}", color_discrete_sequence=[color])
            fig.update_layout(xaxis_title="Índice", yaxis_title=col_name)
        else:
            fig = px.histogram(df, x=col_name, title=f"{col_name}{subtitle}", nbins=30)
        fig.update_layout(margin=dict(t=40, b=30, l=40, r=20), height=350, template='plotly_white')
        return dcc.Graph(figure=fig, config={'responsive': True, 'displayModeBar': True})
    except Exception as e:
        return html.Div([html.H4("❌ Error", className="text-danger"), html.P(str(e))])

@app.callback(
    Output('categorical-chart-output', 'children'),
    Input('generate-categorical-chart', 'n_clicks'),
    [State('categorical-col-select', 'value'),
     State('categorical-chart-type', 'value'),
     State('df-store', 'data'),
     State('categorical-filter-col', 'value'),
     State('categorical-filter-val', 'value')],
    prevent_initial_call=True
)
def generate_categorical_chart(n_clicks, col_name, chart_type, df_data, filter_col, filter_val):
    if not df_data or not col_name:
        return html.Div("⚠️ No hay datos", className="text-warning")
    try:
        df = pd.DataFrame(df_data)
        if col_name not in df.columns:
            return html.Div(f"⚠️ '{col_name}' no existe", className="text-danger")
        filter_info = ""
        if filter_col and filter_val is not None and filter_col in df.columns:
            df = df[df[filter_col] == filter_val]
            if df.empty:
                return html.Div(f"⚠️ No hay registros para {filter_col}='{filter_val}'", className="text-warning")
            filter_info = f"<br><sup>(Filtrado: {filter_col} = '{filter_val}')</sup>"
        freq = df[col_name].value_counts().reset_index()
        freq.columns = [col_name, 'Frecuencia']
        if len(freq) > 20:
            freq = freq.head(20)
            subtitle = f"<br><sup>(Top 20 {filter_info})</sup>"
        else:
            subtitle = f"<br><sup>({len(df)} registros {filter_info})</sup>"
        if chart_type == 'bar':
            fig = px.bar(freq, x=col_name, y='Frecuencia', 
                        title=f"Distribución de {col_name}{subtitle}",
                        color='Frecuencia', color_continuous_scale='Blues', text_auto='.0f')
            fig.update_layout(margin=dict(t=50, b=80, l=40, r=20), height=400, xaxis_tickangle=-45, template='plotly_white')
        elif chart_type == 'pie':
            fig = px.pie(freq, names=col_name, values='Frecuencia',
                        title=f"Distribución de {col_name}{subtitle}",
                        color_discrete_sequence=px.colors.qualitative.Plotly)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=400, template='plotly_white')
        elif chart_type == 'treemap':
            fig = px.treemap(freq, path=[col_name], values='Frecuencia',
                           title=f"Distribución de {col_name}{subtitle}",
                           color='Frecuencia', color_continuous_scale='Blues')
            fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=400, template='plotly_white')
        else:
            fig = px.bar(freq, x=col_name, y='Frecuencia', title=f"{col_name}{subtitle}", text_auto='.0f')
        return dcc.Graph(figure=fig, config={'responsive': True, 'displayModeBar': True})
    except Exception as e:
        return html.Div([html.H4("❌ Error", className="text-danger"), html.P(str(e))])

@app.callback(
    Output('stats-output', 'children'),
    Input('generate-stats', 'n_clicks'),
    [State('stats-col-select', 'value'),
     State('df-store', 'data')],
    prevent_initial_call=True
)
def generate_column_stats(n_clicks, col_name, df_data):
    if not df_data or not col_name:
        return html.Div("⚠️ No hay datos", className="text-warning")
    try:
        df = pd.DataFrame(df_data)
        if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]):
            return html.Div(f"⚠️ '{col_name}' no es numérica", className="text-danger")
        col = df[col_name].dropna()
        stats_data = [
            ['📊 Conteo', f"{len(col):,}"],
            ['📈 Promedio', f"{col.mean():.3f}" if not pd.isna(col.mean()) else "N/A"],
            ['📉 Desv. Estándar', f"{col.std():.3f}" if not pd.isna(col.std()) else "N/A"],
            ['🔻 Mínimo', f"{col.min():.3f}"],
            ['25% (Q1)', f"{col.quantile(0.25):.3f}"],
            ['50% (Mediana)', f"{col.median():.3f}"],
            ['75% (Q3)', f"{col.quantile(0.75):.3f}"],
            ['🔺 Máximo', f"{col.max():.3f}"]
        ]
        metrics = [row[0] for row in stats_data]
        values = [row[1] for row in stats_data]
        fig = go.Figure(data=[go.Table(
            header=dict(values=['<b>Métrica</b>', f'<b>{col_name}</b>'],
                       fill_color='#2563eb', font=dict(color='white', size=14), align='left', height=50),
            cells=dict(values=[metrics, values],
                      fill_color=[['white', '#f8fafc'][i%2] for i in range(len(metrics))],
                      font=dict(size=13), align='left', height=35, line_color='lightgray')
        )])
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=400, template='plotly_white')
        return dcc.Graph(figure=fig, config={'displayModeBar': False, 'responsive': True})
    except Exception as e:
        return html.Div([html.H4("❌ Error", className="text-danger"), html.P(str(e))])

@app.callback(
    [Output('categorical-filter-val', 'options'),
     Output('categorical-filter-val', 'placeholder')],
    Input('categorical-filter-col', 'value'),
    State('df-store', 'data'),
    prevent_initial_call=False
)
def update_filter_values(filter_col, df_data):
    if not filter_col or not df_data:
        return [], "Seleccioná columna filtro primero"
    df = pd.DataFrame(df_data)
    if filter_col not in df.columns:
        return [], "Columna no encontrada"
    vals = df[filter_col].dropna().unique()
    if len(vals) > 100:
        vals = sorted(vals, key=lambda x: str(x))[:100]
    else:
        vals = sorted(vals, key=lambda x: str(x))
    options = [{'label': str(v), 'value': v} for v in vals]
    return options, f"Seleccioná valor ({len(options)} disponibles)"

@app.callback(
    [Output('numeric-filter-val', 'options'),
     Output('numeric-filter-val', 'placeholder')],
    Input('numeric-filter-col', 'value'),
    State('df-store', 'data'),
    prevent_initial_call=False
)
def update_numeric_filter_values(filter_col, df_data):
    if not filter_col or not df_data:
        return [], "Seleccioná columna filtro primero"
    df = pd.DataFrame(df_data)
    if filter_col not in df.columns:
        return [], "Columna no encontrada"
    vals = df[filter_col].dropna().unique()
    if len(vals) > 100:
        vals = sorted(vals, key=lambda x: str(x))[:100]
    else:
        vals = sorted(vals, key=lambda x: str(x))
    options = [{'label': str(v), 'value': v} for v in vals]
    return options, f"Seleccioná valor ({len(options)} disponibles)"

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)