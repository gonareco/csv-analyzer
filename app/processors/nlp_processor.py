import nltk
from textblob import TextBlob
from collections import Counter
import re
import pandas as pd

# Descargar recursos de NLTK (primera ejecución)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class NLPProcessor:
    """Procesador de lenguaje natural para columnas de texto"""
    
    def __init__(self, df, text_column):
        self.df = df
        self.text_column = text_column
        self.stop_words = set(stopwords.words('spanish'))
        
        # Diccionario de sentimientos en español (con pesos)
        self.positive_words = {
            'encanta': 0.8, 'excelente': 1.0, 'bueno': 0.6, 'buena': 0.6, 
            'gusta': 0.5, 'genial': 0.8, 'fantástico': 0.9, 'maravilloso': 0.9,
            'perfecto': 1.0, 'recomiendo': 0.7, 'mejor': 0.7, 'bien': 0.5,
            'feliz': 0.7, 'contento': 0.6, 'satisfecho': 0.7, 'gracias': 0.4,
            'agradecido': 0.6, 'buenísimo': 0.9, 'espectacular': 0.9, 
            'increíble': 0.8, 'hermoso': 0.7, 'lindo': 0.6, 'bonito': 0.6,
            'me encanta': 0.9, 'muy bueno': 0.8, 'muy buena': 0.8
        }
        
        self.negative_words = {
            'odio': -0.8, 'malo': -0.6, 'mala': -0.6, 'pésimo': -1.0,
            'horrible': -0.9, 'terrible': -0.9, 'aburrido': -0.5, 'fatal': -0.8,
            'no me gusta': -0.7, 'no gusta': -0.6, 'decepcionante': -0.8,
            'decepciona': -0.7, 'lamentable': -0.7, 'problema': -0.5,
            'problemas': -0.5, 'queja': -0.6, 'molesto': -0.5, 'muy malo': -0.8,
            'muy mala': -0.8, 'nunca más': -0.6
        }
        
        # NUEVO: Palabras de negación
        self.negation_words = {
            'no', 'nunca', 'jamás', 'tampoco', 'ni', 'sin'
        }
        
        # NUEVO: Intensificadores (multiplicadores)
        self.intensifiers = {
            'muy': 1.5, 'mucho': 1.4, 'demasiado': 1.3, 'realmente': 1.2,
            'absolutamente': 1.3, 'totalmente': 1.2, 'bastante': 1.2,
            'poco': 0.5, 'ligeramente': 0.7
        }
        
        # NUEVO: Conectores de negación compuesta (ej: "no me gusta nada")
        self.negation_phrases = {
            'no me gusta', 'no me gustó', 'no me encanta', 'no me parece'
        }
    
    def analyze_sentiment_es(self, text):
        """
        Manejo de negaciones e intensificadores
        """
        if not isinstance(text, str):
            return 0
        
        text_lower = text.lower()
        
        # 1. Primero buscar frases de negación compuesta
        negation_active = False
        for phrase in self.negation_phrases:
            if phrase in text_lower:
                negation_active = True
                # Reemplazar la frase para no procesarla de nuevo
                text_lower = text_lower.replace(phrase, '')
        
        # 2. Tokenizar para análisis palabra por palabra
        words = word_tokenize(text_lower, language='spanish')
        
        positive_score = 0
        negative_score = 0
        i = 0
        
        while i < len(words):
            word = words[i]
            current_negation = negation_active
            
            # 3. Detectar negación simple
            if word in self.negation_words:
                current_negation = True
                i += 1
                continue
            
            # 4. Detectar intensificador
            intensifier = 1.0
            if word in self.intensifiers:
                intensifier = self.intensifiers[word]
                i += 1
                # Verificar si hay palabra de sentimiento después
                if i < len(words):
                    next_word = words[i]
                    
                    # Buscar en diccionarios
                    if next_word in self.positive_words:
                        score = self.positive_words[next_word] * intensifier
                        if current_negation:
                            score = -score  # Invertir por negación
                        if score > 0:
                            positive_score += score
                        else:
                            negative_score += abs(score)
                        i += 1
                        continue
                    elif next_word in self.negative_words:
                        score = abs(self.negative_words[next_word]) * intensifier
                        if current_negation:
                            score = -score  # "no es malo" = positivo
                        if score > 0:
                            positive_score += score
                        else:
                            negative_score += abs(score)
                        i += 1
                        continue
                continue
            
            # 5. Detectar palabra de sentimiento normal
            if word in self.positive_words:
                score = self.positive_words[word]
                if current_negation:
                    score = -score  # Invertir por negación
                if score > 0:
                    positive_score += score
                else:
                    negative_score += abs(score)
            elif word in self.negative_words:
                score = abs(self.negative_words[word])
                if current_negation:
                    score = -score  # "no es malo" = positivo
                if score > 0:
                    positive_score += score
                else:
                    negative_score += abs(score)
            
            i += 1
        
        # 6. Calcular sentimiento final (-1 a 1)
        total = positive_score + negative_score
        if total == 0:
            return 0
        
        sentiment = (positive_score - negative_score) / total
        return max(-1, min(1, sentiment))  # Limitar entre -1 y 1
    
    def clean_text(self, text):
        """Limpia texto"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def analyze(self):
        """Análisis completo de texto"""
        # Limpiar textos
        texts = self.df[self.text_column].dropna().apply(self.clean_text)
        
        # Contar palabras
        all_words = []
        for text in texts:
            words = nltk.word_tokenize(text)
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        
        # Análisis de sentimiento MEJORADO
        sentiments = []
        for text in self.df[self.text_column].dropna():
            sentiment = self.analyze_sentiment_es(text)  # Usa la versión mejorada
            sentiments.append(sentiment)
        
        # Clasificar sentimientos (umbrales ajustados por la mejora)
        positive = sum(1 for s in sentiments if s > 0.2)  # Reducido de 0.3 a 0.2
        neutral = sum(1 for s in sentiments if -0.2 <= s <= 0.2)
        negative = sum(1 for s in sentiments if s < -0.2)
        total = len(sentiments) if sentiments else 1
        
        return {
            'vocab_size': len(word_counts),
            'top_words': word_counts.most_common(20),
            'wordcloud': word_counts,
            'avg_sentiment': sum(sentiments) / len(sentiments) if sentiments else 0,
            'sentiment_distribution': sentiments,
            'positive_percent': (positive / total) * 100,
            'neutral_percent': (neutral / total) * 100,
            'negative_percent': (negative / total) * 100
        }