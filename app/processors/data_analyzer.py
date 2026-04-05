import pandas as pd
import numpy as np

class DataAnalyzer:
    """Analizador de datos para CSV"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_summary(self):
        """Resumen general del dataset"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in text_cols if self.df[col].nunique() < 20]

        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'numeric_cols': len(numeric_cols),
            'text_cols': len(text_cols),
            'categorical_cols': len(categorical_cols),
            'null_counts': self.df.isnull().sum().sum()
        }

    def get_statistics(self):
        """Estadísticas descriptivas"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return pd.DataFrame()

        stats = numeric_df.describe()
        return stats

    def get_text_columns(self):
        """Identifica columnas que contienen texto"""
        text_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        # Filtrar columnas con texto real (no IDs)
        return [col for col in text_cols if self.df[col].astype(str).str.len().mean() > 10]

    def get_categorical_columns(self):
        """Identifica columnas categóricas"""
        text_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        return [col for col in text_cols if self.df[col].nunique() < 20]
