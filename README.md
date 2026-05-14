![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Dash Version](https://img.shields.io/badge/dash-2.14.0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

> Herramienta web para análisis automático de archivos CSV. Ideal para educación, análisis de datos y exploración rápida de información.

## 🚀 ¿Cómo usarlo?

### Opción 1: Usuarios Generales (sin instalar nada) ⭐ RECOMENDADO

1. Ingresar a https://csv-analyzer-oyp7.onrender.com/ (dale unos 3 minutos a que Render ponga todo en orden para poder usarlo.)
2. Subí tu CSV y empezá a analizar

> Nota: Tu CSV preferentemente debe tener menos de 20000 filas.

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
Demora mucho Render? Tu CSV es quizá muy pesado, truncalo a menos filas y probá.
El wordcloud no aparece	La columna necesita al menos 3 palabras diferentes
Gráfico lento con muchos datos	El sistema muestrea automáticamente (>25k filas)
📄 Licencia

MIT - Libre para uso educativo y comercial.

Agradecimientos a la comunidad de software libre por tanto.

    Plotly Dash

    TextBlob

    Bootstrap

📧 Contacto:
- GitHub: - [gonareco](https://github.com/gonareco)
- LinkedIn: [Gonzalo Areco] www.linkedin.com/in/gonzalo-areco-5b3658223/
- Correo: gonareco+dev@gmail.com

⭐ ¡Si te sirvió, dale una estrella al proyecto!
