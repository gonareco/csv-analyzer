import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

class VizGenerator:
    """Generador de visualizaciones"""
    
    def __init__(self, df):
        self.df = df
    
    def create_stats_table(self, stats_df):
        """Tabla de estadísticas"""
        if stats_df.empty:
            return go.Figure()
        
        stats_rounded = stats_df.round(2)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Estadística'] + list(stats_rounded.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[stats_rounded.index] + [stats_rounded[col] for col in stats_rounded.columns],
                fill_color='lavender',
                align='left'
            )
        )])
        fig.update_layout(title="Estadísticas descriptivas")
        return fig
    
    def create_histograms(self):
        """Histogramas de columnas numéricas en formato cuadrado"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return go.Figure()
        
        from plotly.subplots import make_subplots
        
        # Si hay 1 columna, gráfico cuadrado
        if len(numeric_cols) == 1:
            col = numeric_cols[0]
            fig = px.histogram(self.df, x=col, nbins=30, 
                            color_discrete_sequence=['steelblue'],
                            title=f"Distribución de {col}")
            fig.update_layout(height=500, width=600)  # Más cuadrado
            return fig
        
        # Si hay 2 columnas, 1x2
        elif len(numeric_cols) == 2:
            fig = make_subplots(rows=1, cols=2, 
                            subplot_titles=numeric_cols,
                            horizontal_spacing=0.15)
            for i, col in enumerate(numeric_cols, 1):
                fig.add_trace(go.Histogram(x=self.df[col], nbinsx=30, 
                                        marker_color='steelblue',
                                        name=col), row=1, col=i)
            fig.update_layout(height=450, width=900, 
                            title_text="Distribución de variables numéricas",
                            showlegend=False)
            fig.update_xaxes(title_text="Valor")
            fig.update_yaxes(title_text="Frecuencia")
            return fig
        
        # Si hay más, 2x2 con las primeras 4
        else:
            cols_to_plot = numeric_cols[:4]
            fig = px.histogram(
                self.df.melt(value_vars=cols_to_plot),
                x='value',
                facet_col='variable',
                facet_col_wrap=2,
                title="Distribución de variables numéricas",
                nbins=30,
                color_discrete_sequence=['steelblue']
            )
            fig.update_layout(height=500, width=900)
            return fig
    
    def create_categorical_charts(self):
        """Gráficos para variables categóricas"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if self.df[col].nunique() < 20]
        
        if not categorical_cols:
            return go.Figure()
        
        col = categorical_cols[0]
        counts = self.df[col].value_counts().head(10)
        
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            title=f"Top 10 valores - {col}",
            labels={'x': col, 'y': 'Frecuencia'},
            color=counts.values,
            color_continuous_scale='viridis'
        )
        return fig
    
    def create_wordcloud_plot(self, word_counts):
        """Genera una nube de palabras y retorna la imagen como base64"""
        if not word_counts:
            return None  # ← Retornar None en lugar de Figure vacío
        
        # Configurar backend no-interactive para evitar warnings
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Crear wordcloud con matplotlib
        wc = WordCloud(width=800, height=400, background_color='white', 
                    max_words=50, colormap='viridis')
        wc.generate_from_frequencies(word_counts)
        
        # Convertir a base64
        import io, base64
        img_buffer = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
        img_buffer.seek(0)
        encoded_image = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        return encoded_image  # ← Retornar string base64, no una figura Plotly
        
    def create_sentiment_gauge(self, sentiment_score):
        """Crea un medidor circular de sentimiento"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            title={"text": "Sentimiento promedio", "font": {"size": 16}},
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': " puntos", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [-1, 1], 'tickvals': [-1, -0.5, 0, 0.5, 1],
                         'ticktext': ['Muy negativo', 'Negativo', 'Neutral', 'Positivo', 'Muy positivo'],
                         'tickfont': {'size': 10}},
                'bar': {'color': "#2c3e50", 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-1, -0.6], 'color': "#8B0000", 'name': 'Muy negativo'},
                    {'range': [-0.6, -0.2], 'color': "#CD5C5C", 'name': 'Negativo'},
                    {'range': [-0.2, 0.2], 'color': "#D3D3D3", 'name': 'Neutral'},
                    {'range': [0.2, 0.6], 'color': "#90EE90", 'name': 'Positivo'},
                    {'range': [0.6, 1], 'color': "#006400", 'name': 'Muy positivo'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment_score
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            width=550,
            margin=dict(l=30, r=30, t=50, b=30)
        )
        return fig