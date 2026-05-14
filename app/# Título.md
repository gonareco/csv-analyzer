---
title: "Cómo construí CSV Analyzer: una herramienta web para análisis de datos"
date: 2026-04-18
draft: false
tags: ["Python", "Dash", "Data Science", "Open Source", "Proyectos"]
categories: ["Desarrollo", "Data Science"]
featured: true
---

## ¿Qué es CSV Analyzer?

**CSV Analyzer** es una aplicación web que permite analizar archivos CSV y Excel de forma automática, sin necesidad de escribir código. Nació de una necesidad personal: quería explorar datos rápidamente sin abrir un Jupyter notebook cada vez.

🔗 **Repositorio:** [github.com/gonareco/csv-analyzer](https://github.com/gonareco/csv-analyzer)

---

## Características principales

| Funcionalidad | Descripción |
|---------------|-------------|
| 📊 **Estadísticas automáticas** | Media, mediana, cuartiles, desviación estándar |
| 📈 **Gráficos adaptativos** | Heatmap, scatter, línea, histograma, boxplot, barras |
| 😊 **Análisis de sentimiento** | Detecta emociones en textos + nube de palabras (wordcloud) |
| 🎛️ **Filtros dinámicos** | Exploración interactiva sin recargar |
| 📸 **Exportación PNG** | Guardá tus gráficos como imágenes |
| 🚀 **Manejo de datasets grandes** | Muestreo inteligente para archivos >50k filas |

---

## Tecnologías utilizadas

| Tecnología | Para qué |
|------------|----------|
| **Dash / Plotly** | Interfaz web y gráficos interactivos |
| **Pandas / NumPy** | Manipulación y análisis de datos |
| **TextBlob / NLTK** | Análisis de sentimiento en español/inglés |
| **WordCloud / Matplotlib** | Generación de nubes de palabras |
| **Dash Bootstrap Components** | Estilos y diseño responsivo |

---

## Cómo usarlo (local)

```bash
# Clonar repositorio

git clone https://github.com/gonareco/csv-analyzer.git

# Instalar dependencias
pip install -r requirements.txt

### Ejecutar
python app/app.py

Abrir http://localhost:8050 en tu navegador.

📁 Estructura del proyecto
text

csv-analyzer/
├── app/
│   ├── app.py                 # Aplicación principal
│   └── processors/
│       ├── data_analyzer.py   # Estadísticas
│       ├── nlp_processor.py   # Análisis de texto
│       └── viz_generator.py   # Generación de gráficos
├── requirements.txt
├── README.md
└── LICENSE

Principales desafíos y aprendizajes

1. Manejo de memoria en entornos limitados

Render (plan gratuito) tiene solo 512 MB de RAM. Para procesar CSVs de 10 MB tuve que implementar:

    low_memory=True en pd.read_csv

    Muestreo automático para archivos grandes

    Optimización de tipos de datos (int8, float32, etc.)

2. Detección automática de años en formato ancho

El CSV original venía con columnas año_2015, año_2016, etc. Tuve que implementar una función que detecta y transforma el formato ancho a largo para series temporales.
3. Empaquetado con PyInstaller (y sus limitaciones)

Intenté generar un .exe para Windows, pero las dependencias como nltk y scipy hacían que el ejecutable pesara >200 MB. Aprendí que para una herramienta web, es mejor mantenerla como servicio local.
4. GitHub Actions para CI/CD

Configuré un workflow que compila el proyecto automáticamente al crear un tag. Me llevó varios intentos porque el entorno de Windows en GitHub Actions tiene particularidades con rutas (; vs :) y dependencias nativas.
```
## Cómo usarlo (web)
Sólo hay que ingresar a esta [https://csv-analyzer-oyp7.onrender.com/](url).
Se paciente, al tener un servicio gratuito de Render te obliga a esperar unos segundos hasta que la aplicación está arriba.

Subir un .csv y probar.

>Nota: Estamos limitados por el momento a unas 10000 filas por archivo.

### Lo que aprendí
Dash es potente pero tiene curva de aprendizaje (callbacks, estados, excepciones)
La optimización de memoria es clave cuando trabajás con usuarios que suben archivos grandes.
El logging bien hecho es fundamental para debuggear en producción.
No todo necesita ser un .exe — a veces una aplicación web es más práctica.

### A tener en cuenta:
Dash corre sobre Flask. Para desarrollo usé el servidor integrado, pero para producción (o para permitir múltiples consultas concurrentes) es necesario usar **Gunicorn** (Linux) o **Waitress** (Windows). Esto se configura en Render sin problemas. 

### Próximos pasos (versión 2.0)

1. Gráfico seriado con libertad total de ejes
2. Soporte para series temporales (transformación ancho → largo)
3. Comparador de múltiples CSVs
4. Exportar informes a PDF
5. Mejorar el análisis de sentimiento con transformers (Hugging Face)

### Contribuciones

El proyecto es open source (licencia MIT). Toda contribución es bienvenida.

Repositorio: [github.com/gonareco/csv-analyzer](github.com/gonareco/csv-analyzer)

### Recursos útiles que usé

[https://dash.plotly.com](Documentación Dash)

[www.google.com](Google)

[https://plotly.com/python/plotly-express/](Plotly Express)