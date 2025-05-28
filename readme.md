# Bootcamp en Inteligencia artificial (Talento Tech)
# Nivel: Explorador - Básico-2025-5-L2-G47
# Realizado por: Víctor C. Vladimir Cortés A.
# Descripción del Proyecto:

Analizador de Texto Avanzado, Un sistema completo de análisis de texto en español con múltiples funcionalidades avanzadas.

## Características Principales

- **Análisis de frecuencia** de palabras y n-gramas
- **Análisis de sentimientos** usando múltiples métodos
- **Extracción de entidades nombradas** con spaCy
- **Métricas de legibilidad** y complejidad
- **Visualizaciones interactivas** con gráficos y nubes de palabras
- **Reportes en múltiples formatos** (HTML, JSON, TXT)
- **Análisis comparativo** entre múltiples textos
- **Interfaz de línea de comandos** y modo interactivo

## Instalación Rápida

### Opción 1: Instalación Automática (Recomendada)

```bash
# 1. Descargar todos los archivos del proyecto
# 2. Ejecutar el configurador automático
python setup.py
```

### Opción 2: Instalación Manual

```bash
# 1. Instalar dependencias
pip install nltk pandas numpy matplotlib seaborn wordcloud plotly spacy textblob vaderSentiment

# 2. Descargar modelo de spaCy
python -m spacy download es_core_news_sm

# 3. Verificar instalación
python prueba_rapida.py
```

## Estructura del Proyecto

```
analizador_texto_avanzado/
├── analizador_texto_avanzado.py    # Código principal del analizador
├── ejemplo_uso.py                  # Ejemplo de uso básico
├── analizar_archivo.py            # Interfaz interactiva
├── setup.py                       # Script de instalación
├── prueba_rapida.py               # Verificación del sistema
├── README.md                      # Este archivo
├── textos_ejemplo/                # Archivos de ejemplo
│   ├── inteligencia_artificial.txt
│   ├── cambio_climatico.txt
│   └── educacion_digital.txt
├── resultados/                    # Reportes generados
└── visualizaciones/               # Gráficos y nubes de palabras
```

## Uso del Sistema

### 1. Prueba Rápida

```bash
# Verificar que todo funciona correctamente
python prueba_rapida.py
```

### 2. Análisis Básico

```bash
# Ejemplo con texto predefinido
python ejemplo_uso.py
```

### 3. Análisis Interactivo

```bash
# Interfaz de menú para diferentes opciones
python analizar_archivo.py
```

### 4. Línea de Comandos

```bash
# Analizar un archivo específico
python analizador_texto_avanzado.py --archivo mi_texto.txt

# Analizar todos los archivos de una carpeta
python analizador_texto_avanzado.py --carpeta mis_textos/

# Análisis comparativo de múltiples archivos
python analizador_texto_avanzado.py --carpeta mis_textos/ --comparar

# Generar visualizaciones
python analizador_texto_avanzado.py --archivo mi_texto.txt --visualizaciones

# Especificar formato de reporte
python analizador_texto_avanzado.py --archivo mi_texto.txt --formato json
```

### 5. Uso Programático

```python
from analizador_texto_avanzado import AnalizadorTextoAvanzado

# Crear analizador
analizador = AnalizadorTextoAvanzado()

# Analizar texto
texto = "Tu contenido aquí..."
resultados = analizador.analizar_texto_completo(texto)

# Generar reportes y visualizaciones
analizador.generar_visualizaciones()
analizador.generar_reporte_completo('html')

# Acceder a resultados específicos
print(f"Palabras totales: {resultados['resumen_ejecutivo']['total_palabras']}")
print(f"Sentimiento: {resultados['resumen_ejecutivo']['sentimiento_general']}")
```

## Tipos de Análisis

### Análisis de Palabras
- Frecuencia de palabras individuales
- Bigramas y trigramas más comunes
- Diversidad léxica del texto
- Palabras más y menos frecuentes

### Análisis de Sentimientos
- Polaridad (positivo/negativo/neutral)
- Subjetividad del contenido
- Múltiples algoritmos (TextBlob, VADER)

### Métricas de Legibilidad
- Palabras por oración promedio
- Caracteres por palabra promedio
- Índice de complejidad
- Nivel de dificultad estimado

### Extracción de Entidades
- Personas, lugares, organizaciones
- Fechas y cantidades
- Otros tipos de entidades nombradas

### Análisis Comparativo
- Comparación entre múltiples textos
- Rankings por complejidad y diversidad
- Identificación de temas comunes

## Formatos de Salida

### Reportes
- **HTML**: Visualizable en navegador web
- **JSON**: Datos estructurados para procesamiento
- **TXT**: Texto plano legible

### Visualizaciones
- **Dashboard interactivo**: Gráficos combinados en HTML
- **Nube de palabras**: Imagen PNG
- **Gráficos estadísticos