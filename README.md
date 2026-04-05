
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Dash Version](https://img.shields.io/badge/dash-2.14.0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

> Herramienta web para análisis automático de archivos CSV. Ideal para educación, análisis de datos y exploración rápida de información.

## 🚀 ¿Cómo usarlo?

### Opción 1: Usuarios de Windows (sin instalar nada) ⭐ RECOMENDADO

1. Descargar `CSV_Analyzer.exe` desde [Releases](https://github.com/gonareco/csv-analyzer/releases)
2. Hacer doble clic para ejecutar
3. Se abrirá automáticamente en tu navegador
4. Subí tu CSV y empezá a analizar

> ⚠️ **Para educadores:** Podés distribuir el `.exe` a tus alumnos. No necesitan instalar nada.

### Opción 2: Usuarios de Linux / macOS

```bash
git clone https://github.com/gonareco/csv-analyzer.git
cd csv-analyzer
pip install -r requirements.txt
python app/app.py

Abrir http://localhost:8050 en tu navegador.
Opción 3: Desde código fuente (todos los sistemas)
bash

# Clonar repositorio
git clone https://github.com/gonareco/csv-analyzer.git
cd csv-analyzer

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python app/app.py

✨ Características
Funcionalidad	Descripción
📊 Estadísticas automáticas	Media, mediana, cuartiles, desviación estándar
📈 Gráficos adaptativos	Heatmap, scatter, línea, histograma, boxplot, barras
😊 Análisis de sentimiento	Detecta emociones en textos + nube de palabras
🎛️ Filtros dinámicos	Explorá tus datos sin complicaciones
📸 Exportación PNG	Guardá tus gráficos como imágenes
🚀 Grandes datasets	Maneja archivos de hasta 50MB con muestreo inteligente
🎯 ¿Para quién es?
Usuario	Beneficio
Docentes	Herramienta gratuita para enseñar análisis de datos
Estudiantes	Analizar CSVs sin programar
Investigadores	Exploración rápida de datos
Pequeñas empresas	Análisis de ventas, encuestas, inventarios
📸 Vista previa

    (Agregá capturas de pantalla aquí)

🛠️ Tecnologías

    Dash / Plotly → Interfaz web y gráficos

    Pandas / NumPy → Procesamiento de datos

    TextBlob → Análisis de sentimiento

    WordCloud → Nubes de palabras

📁 Estructura
text

csv-analyzer/
├── app/
│   ├── app.py                 # Aplicación principal
│   └── processors/            # Módulos internos
├── requirements.txt           # Dependencias Python
├── README.md                  # Este archivo
└── CSV_Analyzer.exe           # Versión para Windows (en Releases)

❓ Solución de problemas
Problema	Solución
Windows dice "Protegido"	Clic en "Más info" → "Ejecutar de todas formas"
Error de encoding al subir CSV	Guardá el archivo como UTF-8
El wordcloud no aparece	La columna necesita al menos 3 palabras diferentes
Gráfico lento con muchos datos	El sistema muestrea automáticamente (>50k filas)
📄 Licencia

MIT - Libre para uso educativo y comercial.
🙏 Agradecimientos

    Plotly Dash

    TextBlob

    Bootstrap

📧 Contacto: GitHub

⭐ ¡Si te sirvió, dale una estrella al proyecto!
