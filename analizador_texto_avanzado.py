#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analizador de Texto Avanzado
============================
Sistema completo de análisis de texto con múltiples funcionalidades:
- Análisis de frecuencia de palabras y n-gramas
- Análisis de sentimientos
- Extracción de entidades nombradas
- Visualizaciones interactivas
- Métricas de legibilidad
- Procesamiento de múltiples archivos
"""

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import spacy
from textblob import TextBlob
from collections import Counter, defaultdict
import re
import os
import json
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AnalizadorTextoAvanzado:
    """
    Clase principal para análisis completo de texto en español
    """

    def __init__(self, idioma: str = 'spanish'):
        """
        Inicializa el analizador con configuraciones necesarias

        Args:
            idioma: Idioma para el análisis ('spanish' por defecto)
        """
        self.idioma = idioma
        self.setup_nltk()
        self.setup_spacy()
        self.resultados = {}

    def setup_nltk(self):
        """Configura y descarga recursos necesarios de NLTK"""
        recursos_nltk = [
            'stopwords', 'punkt', 'vader_lexicon',
            'averaged_perceptron_tagger', 'wordnet'
        ]

        for recurso in recursos_nltk:
            try:
                nltk.download(recurso, quiet=True)
            except Exception as e:
                print(f"Warning: No se pudo descargar {recurso}: {e}")

        # Importaciones NLTK
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.probability import FreqDist
        from nltk.util import ngrams
        from nltk.tag import pos_tag

        self.stop_words = set(stopwords.words(self.idioma))
        # Añadir stopwords personalizadas
        stopwords_custom = {
            'así', 'más', 'muy', 'solo', 'también', 'bien', 'vez',
            'hacer', 'ser', 'tener', 'año', 'día', 'vez', 'forma'
        }
        self.stop_words.update(stopwords_custom)

    def setup_spacy(self):
        """Configura spaCy para análisis avanzado"""
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            print("Warning: Modelo de spaCy no encontrado. Instalando...")
            print("Ejecuta: python -m spacy download es_core_news_sm")
            self.nlp = None

    def limpiar_texto(self, texto: str) -> str:
        """
        Limpia y normaliza el texto

        Args:
            texto: Texto a limpiar

        Returns:
            Texto limpio y normalizado
        """
        # Convertir a minúsculas
        texto = texto.lower()

        # Eliminar URLs
        texto = re.sub(r'http\S+|www\S+', '', texto)

        # Eliminar emails
        texto = re.sub(r'\S+@\S+', '', texto)

        # Eliminar números (opcional, mantener fechas importantes)
        texto = re.sub(r'\b\d+\b', '', texto)

        # Normalizar espacios
        texto = re.sub(r'\s+', ' ', texto)

        # Eliminar caracteres especiales excepto puntuación básica
        texto = re.sub(r'[^\w\s.,;:!?¿¡\-]', '', texto)

        return texto.strip()

    def tokenizar_avanzado(self, texto: str) -> Dict[str, List]:
        """
        Tokenización avanzada con diferentes niveles

        Args:
            texto: Texto a tokenizar

        Returns:
            Diccionario con diferentes tipos de tokens
        """
        from nltk.tokenize import word_tokenize, sent_tokenize

        # Tokenización por oraciones
        oraciones = sent_tokenize(texto, language=self.idioma)

        # Tokenización por palabras
        palabras = word_tokenize(texto, language=self.idioma)

        # Filtrar palabras significativas
        palabras_filtradas = [
            palabra for palabra in palabras
            if (palabra.lower() not in self.stop_words and
                palabra.isalpha() and
                len(palabra) > 2)
        ]

        return {
            'oraciones': oraciones,
            'palabras_todas': palabras,
            'palabras_filtradas': palabras_filtradas,
            'total_oraciones': len(oraciones),
            'total_palabras': len(palabras),
            'total_palabras_filtradas': len(palabras_filtradas)
        }

    def analizar_frecuencias(self, palabras: List[str], top_n: int = 20) -> Dict:
        """
        Análisis completo de frecuencias de palabras

        Args:
            palabras: Lista de palabras a analizar
            top_n: Número de palabras más frecuentes a retornar

        Returns:
            Diccionario con análisis de frecuencias
        """
        from nltk.probability import FreqDist

        freq_dist = FreqDist(palabras)

        # Estadísticas básicas
        total_palabras = len(palabras)
        palabras_unicas = len(freq_dist)
        diversidad_lexica = palabras_unicas / total_palabras if total_palabras > 0 else 0

        # Palabras más y menos frecuentes
        mas_frecuentes = freq_dist.most_common(top_n)
        menos_frecuentes = freq_dist.most_common()[-top_n:]

        return {
            'distribucion_frecuencias': freq_dist,
            'total_palabras': total_palabras,
            'palabras_unicas': palabras_unicas,
            'diversidad_lexica': round(diversidad_lexica, 4),
            'mas_frecuentes': mas_frecuentes,
            'menos_frecuentes': menos_frecuentes,
            'frecuencia_promedio': np.mean(list(freq_dist.values()))
        }

    def analizar_ngramas(self, palabras: List[str], n: int = 2, top_n: int = 10) -> List[Tuple]:
        """
        Análisis de n-gramas (bigramas, trigramas, etc.)

        Args:
            palabras: Lista de palabras
            n: Tamaño del n-grama
            top_n: Número de n-gramas más frecuentes

        Returns:
            Lista de n-gramas más frecuentes
        """
        from nltk.util import ngrams

        ngramas = list(ngrams(palabras, n))
        freq_ngramas = Counter(ngramas)

        return freq_ngramas.most_common(top_n)

    def analizar_sentimientos(self, texto: str) -> Dict:
        """
        Análisis de sentimientos usando múltiples métodos

        Args:
            texto: Texto a analizar

        Returns:
            Diccionario con análisis de sentimientos
        """
        resultados = {}

        # TextBlob (traducido al inglés para mejor precisión)
        try:
            blob = TextBlob(texto)
            # Traducir al inglés para mejor análisis
            texto_en = str(blob.translate(to='en'))
            blob_en = TextBlob(texto_en)

            resultados['textblob'] = {
                'polaridad': round(blob_en.sentiment.polarity, 4),
                'subjetividad': round(blob_en.sentiment.subjectivity, 4),
                'clasificacion': self._clasificar_sentimiento(blob_en.sentiment.polarity)
            }
        except Exception as e:
            print(f"Error en análisis TextBlob: {e}")

        # VADER (inglés)
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(texto_en)

            resultados['vader'] = {
                'compuesto': round(scores['compound'], 4),
                'positivo': round(scores['pos'], 4),
                'neutral': round(scores['neu'], 4),
                'negativo': round(scores['neg'], 4),
                'clasificacion': self._clasificar_sentimiento(scores['compound'])
            }
        except Exception as e:
            print(f"Error en análisis VADER: {e}")

        return resultados

    def _clasificar_sentimiento(self, score: float) -> str:
        """Clasifica el sentimiento basado en el score"""
        if score >= 0.1:
            return 'Positivo'
        elif score <= -0.1:
            return 'Negativo'
        else:
            return 'Neutral'

    def extraer_entidades(self, texto: str) -> Dict:
        """
        Extracción de entidades nombradas usando spaCy

        Args:
            texto: Texto para extraer entidades

        Returns:
            Diccionario con entidades encontradas
        """
        if not self.nlp:
            return {'error': 'Modelo spaCy no disponible'}

        doc = self.nlp(texto)

        entidades = {}
        for ent in doc.ents:
            tipo = ent.label_
            if tipo not in entidades:
                entidades[tipo] = []
            entidades[tipo].append({
                'texto': ent.text,
                'inicio': ent.start_char,
                'fin': ent.end_char
            })

        # Estadísticas de entidades
        resumen_entidades = {tipo: len(lista) for tipo, lista in entidades.items()}

        return {
            'entidades_detalladas': entidades,
            'resumen': resumen_entidades,
            'total_entidades': sum(resumen_entidades.values())
        }

    def calcular_metricas_legibilidad(self, texto: str) -> Dict:
        """
        Calcula métricas de legibilidad del texto

        Args:
            texto: Texto a analizar

        Returns:
            Diccionario con métricas de legibilidad
        """
        from nltk.tokenize import sent_tokenize, word_tokenize

        oraciones = sent_tokenize(texto, language=self.idioma)
        palabras = word_tokenize(texto, language=self.idioma)

        # Métricas básicas
        num_oraciones = len(oraciones)
        num_palabras = len([p for p in palabras if p.isalpha()])

        if num_oraciones == 0 or num_palabras == 0:
            return {'error': 'Texto insuficiente para análisis'}

        # Longitud promedio de oraciones
        palabras_por_oracion = num_palabras / num_oraciones

        # Longitud promedio de palabras
        caracteres_por_palabra = np.mean([len(p) for p in palabras if p.isalpha()])

        # Índice de complejidad (simplificado)
        indice_complejidad = (palabras_por_oracion * 0.4) + (caracteres_por_palabra * 2.5)

        # Clasificación de dificultad
        if indice_complejidad < 30:
            dificultad = 'Muy fácil'
        elif indice_complejidad < 50:
            dificultad = 'Fácil'
        elif indice_complejidad < 70:
            dificultad = 'Moderado'
        elif indice_complejidad < 90:
            dificultad = 'Difícil'
        else:
            dificultad = 'Muy difícil'

        return {
            'oraciones': num_oraciones,
            'palabras': num_palabras,
            'palabras_por_oracion': round(palabras_por_oracion, 2),
            'caracteres_por_palabra': round(caracteres_por_palabra, 2),
            'indice_complejidad': round(indice_complejidad, 2),
            'nivel_dificultad': dificultad
        }

    def analizar_texto_completo(self, texto: str) -> Dict:
        """
        Realiza análisis completo del texto

        Args:
            texto: Texto a analizar

        Returns:
            Diccionario con todos los análisis
        """
        print("Iniciando análisis completo del texto...")

        # Limpiar texto
        texto_limpio = self.limpiar_texto(texto)

        # Tokenización
        print("Tokenizando texto...")
        tokens = self.tokenizar_avanzado(texto_limpio)

        # Análisis de frecuencias
        print("Analizando frecuencias...")
        frecuencias = self.analizar_frecuencias(tokens['palabras_filtradas'])

        # N-gramas
        print("Analizando bigramas y trigramas...")
        bigramas = self.analizar_ngramas(tokens['palabras_filtradas'], n=2)
        trigramas = self.analizar_ngramas(tokens['palabras_filtradas'], n=3)

        # Análisis de sentimientos
        print("Analizando sentimientos...")
        sentimientos = self.analizar_sentimientos(texto)

        # Extracción de entidades
        print("Extrayendo entidades...")
        entidades = self.extraer_entidades(texto)

        # Métricas de legibilidad
        print("Calculando legibilidad...")
        legibilidad = self.calcular_metricas_legibilidad(texto)

        # Compilar resultados
        self.resultados = {
            'texto_original': texto,
            'texto_limpio': texto_limpio,
            'tokenizacion': tokens,
            'frecuencias': frecuencias,
            'bigramas': bigramas,
            'trigramas': trigramas,
            'sentimientos': sentimientos,
            'entidades': entidades,
            'legibilidad': legibilidad,
            'resumen_ejecutivo': self._generar_resumen_ejecutivo()
        }

        print("Análisis completado!")
        return self.resultados

    def _generar_resumen_ejecutivo(self) -> Dict:
        """Genera un resumen ejecutivo de los resultados"""
        if not hasattr(self, 'resultados') or not self.resultados:
            # Usar datos temporales durante la construcción
            return {
                'total_palabras': 0,
                'palabras_unicas': 0,
                'diversidad_lexica': 0,
                'sentimiento_general': 'No determinado',
                'complejidad': 'N/A',
                'temas_principales': []
            }

        return {
            'total_palabras': self.resultados.get('tokenizacion', {}).get('total_palabras', 0),
            'palabras_unicas': len(set(self.resultados.get('tokenizacion', {}).get('palabras_filtradas', []))),
            'diversidad_lexica': self.resultados.get('frecuencias', {}).get('diversidad_lexica', 0),
            'sentimiento_general': self._obtener_sentimiento_predominante(),
            'complejidad': self.resultados.get('legibilidad', {}).get('nivel_dificultad', 'N/A'),
            'temas_principales': [palabra for palabra, freq in
                                  self.resultados.get('frecuencias', {}).get('mas_frecuentes', [])[:5]]
        }

    def _obtener_sentimiento_predominante(self) -> str:
        """Obtiene el sentimiento predominante del texto"""
        if not hasattr(self, 'resultados') or not self.resultados:
            return 'No determinado'

        sentimientos = self.resultados.get('sentimientos', {})
        if 'textblob' in sentimientos:
            return sentimientos['textblob']['clasificacion']
        elif 'vader' in sentimientos:
            return sentimientos['vader']['clasificacion']
        return 'No determinado'

    def generar_visualizaciones(self, guardar: bool = True):
        """
        Genera visualizaciones completas de los resultados

        Args:
            guardar: Si guardar las visualizaciones en archivos
        """
        if not self.resultados:
            print("Error: No hay resultados para visualizar. Ejecuta primero el análisis.")
            return

        print("Generando visualizaciones...")

        # Configurar subplot
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Top 15 Palabras Más Frecuentes',
                'Distribución de Sentimientos',
                'Longitud de Oraciones',
                'Bigramas Más Comunes',
                'Nube de Palabras',
                'Métricas de Complejidad'
            ],
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"colspan": 2}, None],
                [{"type": "indicator"}, None]
            ]
        )

        # 1. Palabras más frecuentes
        freq_data = self.resultados['frecuencias']['mas_frecuentes'][:15]
        palabras = [item[0] for item in freq_data]
        frecuencias = [item[1] for item in freq_data]

        fig.add_trace(
            go.Bar(x=frecuencias, y=palabras, orientation='h', name='Frecuencia'),
            row=1, col=1
        )

        # 2. Sentimientos
        if 'textblob' in self.resultados['sentimientos']:
            sent_data = self.resultados['sentimientos']['textblob']
            fig.add_trace(
                go.Pie(
                    labels=['Positivo', 'Neutral', 'Negativo'],
                    values=[
                        max(0, sent_data['polaridad']),
                        1 - abs(sent_data['polaridad']),
                        max(0, -sent_data['polaridad'])
                    ]
                ),
                row=1, col=2
            )

        # 3. Longitud de oraciones
        oraciones = self.resultados['tokenizacion']['oraciones']
        longitudes = [len(oracion.split()) for oracion in oraciones]

        fig.add_trace(
            go.Histogram(x=longitudes, nbinsx=20, name='Distribución'),
            row=2, col=1
        )

        # 4. Bigramas
        bigramas_data = self.resultados['bigramas'][:10]
        if bigramas_data:
            bigrama_labels = [' '.join(bigrama[0]) for bigrama in bigramas_data]
            bigrama_values = [bigrama[1] for bigrama in bigramas_data]

            fig.add_trace(
                go.Bar(x=bigrama_values, y=bigrama_labels, orientation='h'),
                row=2, col=2
            )

        # Actualizar layout
        fig.update_layout(
            height=1200,
            title_text="Dashboard de Análisis de Texto",
            showlegend=False
        )

        # Mostrar gráfico
        fig.show()

        # Generar nube de palabras
        self._generar_nube_palabras(guardar)

        if guardar:
            fig.write_html("dashboard_analisis_texto.html")
            print("Visualizaciones guardadas en 'dashboard_analisis_texto.html'")

    def _generar_nube_palabras(self, guardar: bool = True):
        """Genera nube de palabras"""
        palabras_filtradas = self.resultados['tokenizacion']['palabras_filtradas']
        texto_nube = ' '.join(palabras_filtradas)

        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(texto_nube)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nube de Palabras', fontsize=16, fontweight='bold')

        if guardar:
            plt.savefig('nube_palabras.png', dpi=300, bbox_inches='tight')
            print("Nube de palabras guardada en 'nube_palabras.png'")

        plt.show()

    def generar_reporte_completo(self, formato: str = 'html') -> str:
        """
        Genera un reporte completo en formato especificado

        Args:
            formato: Formato del reporte ('html', 'json', 'txt')

        Returns:
            Ruta del archivo generado
        """
        if not self.resultados:
            print("Error: No hay resultados para generar reporte.")
            return ""

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        if formato == 'html':
            return self._generar_reporte_html(timestamp)
        elif formato == 'json':
            return self._generar_reporte_json(timestamp)
        elif formato == 'txt':
            return self._generar_reporte_txt(timestamp)
        else:
            print("Formato no soportado. Usa 'html', 'json' o 'txt'")
            return ""

    def _generar_reporte_html(self, timestamp: str) -> str:
        """Genera reporte en formato HTML"""
        filename = f"reporte_analisis_{timestamp}.html"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Análisis de Texto</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Reporte de Análisis de Texto</h1>
                <p><strong>Fecha:</strong> {pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")}</p>
                <p><strong>Longitud del texto:</strong> {len(self.resultados['texto_original'])} caracteres</p>
            </div>

            <div class="section">
                <h2>Resumen Ejecutivo</h2>
                <div class="metric">
                    <strong>Palabras totales:</strong> {self.resultados['resumen_ejecutivo']['total_palabras']}
                </div>
                <div class="metric">
                    <strong>Palabras únicas:</strong> {self.resultados['resumen_ejecutivo']['palabras_unicas']}
                </div>
                <div class="metric">
                    <strong>Diversidad léxica:</strong> {self.resultados['resumen_ejecutivo']['diversidad_lexica']}
                </div>
                <div class="metric">
                    <strong>Sentimiento:</strong> {self.resultados['resumen_ejecutivo']['sentimiento_general']}
                </div>
                <div class="metric">
                    <strong>Complejidad:</strong> {self.resultados['resumen_ejecutivo']['complejidad']}
                </div>
            </div>

            <div class="section">
                <h2>Análisis de Frecuencias</h2>
                <h3>Top 10 Palabras Más Frecuentes</h3>
                <table>
                    <tr><th>Palabra</th><th>Frecuencia</th></tr>
        """

        for palabra, freq in self.resultados['frecuencias']['mas_frecuentes'][:10]:
            html_content += f"<tr><td>{palabra}</td><td>{freq}</td></tr>"

        html_content += """
                </table>
            </div>

            <div class="section">
                <h2>Análisis de Sentimientos</h2>
        """

        if 'textblob' in self.resultados['sentimientos']:
            sent = self.resultados['sentimientos']['textblob']
            html_content += f"""
                <p><strong>Polaridad:</strong> {sent['polaridad']} ({sent['clasificacion']})</p>
                <p><strong>Subjetividad:</strong> {sent['subjetividad']}</p>
            """

        html_content += """
            </div>

            <div class="section">
                <h2>Métricas de Legibilidad</h2>
        """

        leg = self.resultados['legibilidad']
        html_content += f"""
                <p><strong>Oraciones:</strong> {leg.get('oraciones', 'N/A')}</p>
                <p><strong>Palabras por oración:</strong> {leg.get('palabras_por_oracion', 'N/A')}</p>
                <p><strong>Caracteres por palabra:</strong> {leg.get('caracteres_por_palabra', 'N/A')}</p>
                <p><strong>Nivel de dificultad:</strong> {leg.get('nivel_dificultad', 'N/A')}</p>
        """

        html_content += """
            </div>
        </body>
        </html>
        """

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Reporte HTML generado: {filename}")
        return filename

    def _generar_reporte_json(self, timestamp: str) -> str:
        """Genera reporte en formato JSON"""
        filename = f"reporte_analisis_{timestamp}.json"

        # Limpiar resultados para JSON
        resultados_json = {}
        for key, value in self.resultados.items():
            if key != 'frecuencias':  # FreqDist no es serializable
                resultados_json[key] = value
            else:
                resultados_json[key] = {
                    'mas_frecuentes': value['mas_frecuentes'],
                    'total_palabras': value['total_palabras'],
                    'palabras_unicas': value['palabras_unicas'],
                    'diversidad_lexica': value['diversidad_lexica']
                }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(resultados_json, f, ensure_ascii=False, indent=2)

        print(f"Reporte JSON generado: {filename}")
        return filename

    def _generar_reporte_txt(self, timestamp: str) -> str:
        """Genera reporte en formato texto plano"""
        filename = f"reporte_analisis_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("REPORTE DE ANÁLISIS DE TEXTO\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Fecha: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"Longitud del texto: {len(self.resultados['texto_original'])} caracteres\n\n")

            # Resumen ejecutivo
            f.write("RESUMEN EJECUTIVO\n")
            f.write("-" * 20 + "\n")
            resumen = self.resultados['resumen_ejecutivo']
            for key, value in resumen.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")

            # Palabras más frecuentes
            f.write("TOP 15 PALABRAS MÁS FRECUENTES\n")
            f.write("-" * 35 + "\n")
            for i, (palabra, freq) in enumerate(self.resultados['frecuencias']['mas_frecuentes'][:15], 1):
                f.write(f"{i:2d}. {palabra:<20} {freq:>3d}\n")
            f.write("\n")

            # Sentimientos
            f.write("ANÁLISIS DE SENTIMIENTOS\n")
            f.write("-" * 25 + "\n")
            if 'textblob' in self.resultados['sentimientos']:
                sent = self.resultados['sentimientos']['textblob']
                f.write(f"Polaridad: {sent['polaridad']} ({sent['clasificacion']})\n")
                f.write(f"Subjetividad: {sent['subjetividad']}\n")
            f.write("\n")

            # Legibilidad
            f.write("MÉTRICAS DE LEGIBILIDAD\n")
            f.write("-" * 25 + "\n")
            leg = self.resultados['legibilidad']
            for key, value in leg.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")

            # Bigramas
            f.write("TOP 10 BIGRAMAS\n")
            f.write("-" * 15 + "\n")
            for i, (bigrama, freq) in enumerate(self.resultados['bigramas'][:10], 1):
                f.write(f"{i:2d}. {' '.join(bigrama):<25} {freq:>3d}\n")

        print(f"Reporte TXT generado: {filename}")
        return filename


def procesar_texto_desde_archivo(ruta_archivo: str) -> Optional[str]:
    """
    Lee y procesa un archivo de texto

    Args:
        ruta_archivo: Ruta al archivo de texto

    Returns:
        Contenido del archivo o None si hay error
    """
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            return archivo.read()
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado: {ruta_archivo}")
        return None
    except Exception as e:
        print(f"Error al leer archivo: {e}")
        return None


def procesar_multiples_archivos(carpeta: str) -> Dict[str, Dict]:
    """
    Procesa múltiples archivos de texto en una carpeta

    Args:
        carpeta: Ruta de la carpeta con archivos

    Returns:
        Diccionario con resultados de cada archivo
    """
    if not os.path.exists(carpeta):
        print(f"Error: Carpeta no encontrada: {carpeta}")
        return {}

    extensiones_validas = {'.txt', '.md', '.rst', '.text'}
    archivos_procesados = {}

    analizador = AnalizadorTextoAvanzado()

    for nombre_archivo in os.listdir(carpeta):
        ruta_completa = os.path.join(carpeta, nombre_archivo)

        if (os.path.isfile(ruta_completa) and
                any(nombre_archivo.lower().endswith(ext) for ext in extensiones_validas)):

            print(f"Procesando: {nombre_archivo}")

            texto = procesar_texto_desde_archivo(ruta_completa)
            if texto:
                try:
                    resultados = analizador.analizar_texto_completo(texto)
                    archivos_procesados[nombre_archivo] = resultados
                    print(f"Completado: {nombre_archivo}")
                except Exception as e:
                    print(f"Error procesando {nombre_archivo}: {e}")

    return archivos_procesados


def comparar_textos(textos: Dict[str, str]) -> Dict:
    """
    Compara múltiples textos y genera un análisis comparativo

    Args:
        textos: Diccionario con nombre del texto como clave y contenido como valor

    Returns:
        Diccionario con análisis comparativo
    """
    analizador = AnalizadorTextoAvanzado()
    resultados_comparacion = {}

    print("Iniciando análisis comparativo...")

    # Analizar cada texto
    for nombre, texto in textos.items():
        print(f"Analizando: {nombre}")
        resultados_comparacion[nombre] = analizador.analizar_texto_completo(texto)

    # Generar comparación
    comparacion = {
        'resumen_comparativo': {},
        'metricas_por_texto': {},
        'ranking_complejidad': [],
        'ranking_diversidad': [],
        'palabras_comunes': set()
    }

    # Extraer métricas de cada texto
    for nombre, resultado in resultados_comparacion.items():
        resumen = resultado['resumen_ejecutivo']
        comparacion['metricas_por_texto'][nombre] = {
            'palabras_totales': resumen['total_palabras'],
            'diversidad_lexica': resumen['diversidad_lexica'],
            'sentimiento': resumen['sentimiento_general'],
            'complejidad': resumen['complejidad']
        }

    # Crear rankings
    textos_ordenados = list(comparacion['metricas_por_texto'].items())

    # Ranking por complejidad (mapear niveles a números)
    niveles_complejidad = {
        'Muy fácil': 1, 'Fácil': 2, 'Moderado': 3,
        'Difícil': 4, 'Muy difícil': 5, 'N/A': 0
    }

    comparacion['ranking_complejidad'] = sorted(
        textos_ordenados,
        key=lambda x: niveles_complejidad.get(x[1]['complejidad'], 0),
        reverse=True
    )

    # Ranking por diversidad léxica
    comparacion['ranking_diversidad'] = sorted(
        textos_ordenados,
        key=lambda x: x[1]['diversidad_lexica'],
        reverse=True
    )

    # Encontrar palabras comunes
    todas_las_palabras = []
    for resultado in resultados_comparacion.values():
        palabras_frecuentes = [palabra for palabra, _ in resultado['frecuencias']['mas_frecuentes'][:10]]
        todas_las_palabras.extend(palabras_frecuentes)

    contador_palabras = Counter(todas_las_palabras)
    comparacion['palabras_comunes'] = [
        palabra for palabra, freq in contador_palabras.items()
        if freq >= len(textos) / 2  # Aparece en al menos la mitad de los textos
    ]

    return {
        'resultados_individuales': resultados_comparacion,
        'analisis_comparativo': comparacion
    }


def generar_reporte_comparativo(resultados_comparacion: Dict, formato: str = 'html') -> str:
    """
    Genera un reporte comparativo de múltiples textos

    Args:
        resultados_comparacion: Resultados del análisis comparativo
        formato: Formato del reporte ('html', 'txt', 'json')

    Returns:
        Ruta del archivo generado
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    if formato == 'html':
        return _generar_reporte_comparativo_html(resultados_comparacion, timestamp)
    elif formato == 'txt':
        return _generar_reporte_comparativo_txt(resultados_comparacion, timestamp)
    elif formato == 'json':
        return _generar_reporte_comparativo_json(resultados_comparacion, timestamp)
    else:
        print("Formato no soportado. Usa 'html', 'txt' o 'json'")
        return ""


def _generar_reporte_comparativo_html(resultados: Dict, timestamp: str) -> str:
    """Genera reporte comparativo en HTML"""
    filename = f"reporte_comparativo_{timestamp}.html"

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte Comparativo de Textos</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            .section { margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .ranking { background-color: #f9f9f9; padding: 10px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Reporte Comparativo de Análisis de Textos</h1>
            <p><strong>Fecha:</strong> """ + pd.Timestamp.now().strftime("%d/%m/%Y %H:%M") + """</p>
        </div>

        <div class="section">
            <h2>Resumen Comparativo</h2>
            <table>
                <tr>
                    <th>Texto</th>
                    <th>Palabras Totales</th>
                    <th>Diversidad Léxica</th>
                    <th>Sentimiento</th>
                    <th>Complejidad</th>
                </tr>
    """

    # Añadir filas de la tabla
    for nombre, metricas in resultados['analisis_comparativo']['metricas_por_texto'].items():
        html_content += f"""
                <tr>
                    <td>{nombre}</td>
                    <td>{metricas['palabras_totales']}</td>
                    <td>{metricas['diversidad_lexica']:.4f}</td>
                    <td>{metricas['sentimiento']}</td>
                    <td>{metricas['complejidad']}</td>
                </tr>
        """

    html_content += """
            </table>
        </div>

        <div class="section">
            <h2>Rankings</h2>
            <div class="ranking">
                <h3>Ranking por Complejidad (Mayor a Menor)</h3>
                <ol>
    """

    # Ranking de complejidad
    for nombre, metricas in resultados['analisis_comparativo']['ranking_complejidad']:
        html_content += f"<li>{nombre} - {metricas['complejidad']}</li>"

    html_content += """
                </ol>
            </div>

            <div class="ranking">
                <h3>Ranking por Diversidad Léxica (Mayor a Menor)</h3>
                <ol>
    """

    # Ranking de diversidad
    for nombre, metricas in resultados['analisis_comparativo']['ranking_diversidad']:
        html_content += f"<li>{nombre} - {metricas['diversidad_lexica']:.4f}</li>"

    html_content += """
                </ol>
            </div>
        </div>

        <div class="section">
            <h2>Palabras Comunes</h2>
            <p>Palabras que aparecen frecuentemente en múltiples textos:</p>
            <p>""" + ", ".join(resultados['analisis_comparativo']['palabras_comunes']) + """</p>
        </div>
    </body>
    </html>
    """

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Reporte comparativo HTML generado: {filename}")
    return filename


def _generar_reporte_comparativo_txt(resultados: Dict, timestamp: str) -> str:
    """Genera reporte comparativo en TXT"""
    filename = f"reporte_comparativo_{timestamp}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE COMPARATIVO DE ANÁLISIS DE TEXTOS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Fecha: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write(f"Textos analizados: {len(resultados['analisis_comparativo']['metricas_por_texto'])}\n\n")

        # Resumen por texto
        f.write("RESUMEN POR TEXTO\n")
        f.write("-" * 20 + "\n")
        for nombre, metricas in resultados['analisis_comparativo']['metricas_por_texto'].items():
            f.write(f"\n{nombre}:\n")
            f.write(f"  Palabras totales: {metricas['palabras_totales']}\n")
            f.write(f"  Diversidad léxica: {metricas['diversidad_lexica']:.4f}\n")
            f.write(f"  Sentimiento: {metricas['sentimiento']}\n")
            f.write(f"  Complejidad: {metricas['complejidad']}\n")

        # Rankings
        f.write("\n\nRANKING POR COMPLEJIDAD\n")
        f.write("-" * 25 + "\n")
        for i, (nombre, metricas) in enumerate(resultados['analisis_comparativo']['ranking_complejidad'], 1):
            f.write(f"{i}. {nombre} - {metricas['complejidad']}\n")

        f.write("\n\nRANKING POR DIVERSIDAD LÉXICA\n")
        f.write("-" * 30 + "\n")
        for i, (nombre, metricas) in enumerate(resultados['analisis_comparativo']['ranking_diversidad'], 1):
            f.write(f"{i}. {nombre} - {metricas['diversidad_lexica']:.4f}\n")

        # Palabras comunes
        f.write("\n\nPALABRAS COMUNES\n")
        f.write("-" * 15 + "\n")
        palabras_comunes = resultados['analisis_comparativo']['palabras_comunes']
        if palabras_comunes:
            f.write(", ".join(palabras_comunes) + "\n")
        else:
            f.write("No se encontraron palabras comunes significativas.\n")

    print(f"Reporte comparativo TXT generado: {filename}")
    return filename


def _generar_reporte_comparativo_json(resultados: Dict, timestamp: str) -> str:
    """Genera reporte comparativo en JSON"""
    filename = f"reporte_comparativo_{timestamp}.json"

    # Limpiar datos para JSON
    datos_json = {
        'metadatos': {
            'fecha_generacion': pd.Timestamp.now().isoformat(),
            'textos_analizados': len(resultados['analisis_comparativo']['metricas_por_texto'])
        },
        'analisis_comparativo': resultados['analisis_comparativo']
    }

    # Convertir conjuntos a listas para JSON
    if isinstance(datos_json['analisis_comparativo']['palabras_comunes'], set):
        datos_json['analisis_comparativo']['palabras_comunes'] = list(
            datos_json['analisis_comparativo']['palabras_comunes']
        )

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(datos_json, f, ensure_ascii=False, indent=2)

    print(f"Reporte comparativo JSON generado: {filename}")
    return filename


def main():
    """Función principal para ejecutar el analizador desde línea de comandos"""
    parser = argparse.ArgumentParser(description='Analizador de Texto Avanzado')
    parser.add_argument('--archivo', type=str, help='Archivo de texto a analizar')
    parser.add_argument('--carpeta', type=str, help='Carpeta con archivos de texto')
    parser.add_argument('--formato', type=str, default='html',
                        choices=['html', 'json', 'txt'], help='Formato del reporte')
    parser.add_argument('--visualizaciones', action='store_true',
                        help='Generar visualizaciones')
    parser.add_argument('--comparar', action='store_true',
                        help='Modo comparativo para múltiples archivos')

    args = parser.parse_args()

    if args.archivo:
        # Analizar un archivo específico
        texto = procesar_texto_desde_archivo(args.archivo)
        if texto:
            analizador = AnalizadorTextoAvanzado()
            resultados = analizador.analizar_texto_completo(texto)

            if args.visualizaciones:
                analizador.generar_visualizaciones()

            analizador.generar_reporte_completo(args.formato)

    elif args.carpeta:
        if args.comparar:
            # Modo comparativo
            archivos_resultado = procesar_multiples_archivos(args.carpeta)
            if archivos_resultado:
                textos_contenido = {}
                for nombre_archivo in archivos_resultado:
                    ruta_completa = os.path.join(args.carpeta, nombre_archivo)
                    contenido = procesar_texto_desde_archivo(ruta_completa)
                    if contenido:
                        textos_contenido[nombre_archivo] = contenido

                if textos_contenido:
                    resultados_comparacion = comparar_textos(textos_contenido)
                    generar_reporte_comparativo(resultados_comparacion, args.formato)
        else:
            # Procesar archivos individualmente
            procesar_multiples_archivos(args.carpeta)

    else:
        # Modo interactivo
        print("Analizador de Texto Avanzado")
        print("=" * 40)
        print("1. Analizar texto directo")
        print("2. Analizar archivo")
        print("3. Analizar carpeta")
        print("4. Comparar múltiples textos")

        opcion = input("\nSelecciona una opción (1-4): ").strip()

        if opcion == "1":
            print("Ingresa tu texto (Ctrl+D para terminar):")
            texto = ""
            try:
                while True:
                    linea = input()
                    texto += linea + "\n"
            except EOFError:
                pass

            if texto.strip():
                analizador = AnalizadorTextoAvanzado()
                analizador.analizar_texto_completo(texto)
                analizador.generar_visualizaciones()
                analizador.generar_reporte_completo('html')

        elif opcion == "2":
            archivo = input("Ruta del archivo: ").strip()
            texto = procesar_texto_desde_archivo(archivo)
            if texto:
                analizador = AnalizadorTextoAvanzado()
                analizador.analizar_texto_completo(texto)
                analizador.generar_visualizaciones()
                analizador.generar_reporte_completo('html')

        elif opcion == "3":
            carpeta = input("Ruta de la carpeta: ").strip()
            procesar_multiples_archivos(carpeta)

        elif opcion == "4":
            carpeta = input("Ruta de la carpeta con textos a comparar: ").strip()
            archivos_resultado = procesar_multiples_archivos(carpeta)
            if archivos_resultado:
                textos_contenido = {}
                for nombre_archivo in archivos_resultado:
                    ruta_completa = os.path.join(carpeta, nombre_archivo)
                    contenido = procesar_texto_desde_archivo(ruta_completa)
                    if contenido:
                        textos_contenido[nombre_archivo] = contenido

                if textos_contenido:
                    resultados_comparacion = comparar_textos(textos_contenido)
                    generar_reporte_comparativo(resultados_comparacion, 'html')


if __name__ == "__main__":
    main()