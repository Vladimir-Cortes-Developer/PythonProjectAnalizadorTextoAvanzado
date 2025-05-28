#!/usr/bin/env python3
"""
Setup y Configuración del Analizador de Texto Avanzado - Versión Corregida
=========================================================================

Este script configura el entorno y resuelve problemas comunes del analizador.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path


def instalar_paquete(paquete, nombre_mostrar=None):
    """
    Instala un paquete usando pip

    Args:
        paquete: Nombre del paquete a instalar
        nombre_mostrar: Nombre para mostrar (opcional)
    """
    if nombre_mostrar is None:
        nombre_mostrar = paquete

    try:
        print(f"Instalando {nombre_mostrar}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", paquete])
        print(f"✓ {nombre_mostrar} instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error instalando {nombre_mostrar}: {e}")
        return False


def verificar_e_instalar_dependencias():
    """
    Verifica e instala las dependencias necesarias
    """
    print("=" * 60)
    print("VERIFICACIÓN E INSTALACIÓN DE DEPENDENCIAS")
    print("=" * 60)

    dependencias = [
        ("nltk", "Natural Language Toolkit"),
        ("pandas", "Pandas - Data Analysis"),
        ("numpy", "NumPy - Numerical Computing"),
        ("matplotlib", "Matplotlib - Plotting"),
        ("seaborn", "Seaborn - Statistical Visualization"),
        ("wordcloud", "WordCloud - Nube de Palabras"),
        ("plotly", "Plotly - Gráficos Interactivos"),
        ("textblob", "TextBlob - Procesamiento de Texto"),
        ("spacy", "spaCy - NLP Avanzado"),
        ("vaderSentiment", "VADER - Análisis de Sentimientos")
    ]

    instalados = 0
    fallos = []

    for paquete, descripcion in dependencias:
        try:
            importlib.import_module(paquete.split('==')[0])
            print(f"✓ {paquete:<15} - Ya instalado - {descripcion}")
            instalados += 1
        except ImportError:
            print(f"⚠ {paquete:<15} - No encontrado, instalando...")
            if instalar_paquete(paquete, descripcion):
                instalados += 1
            else:
                fallos.append(paquete)

    print(f"\nResumen: {instalados}/{len(dependencias)} dependencias instaladas")

    if fallos:
        print(f"Paquetes con fallos: {', '.join(fallos)}")

    return len(fallos) == 0


def configurar_nltk():
    """
    Configura y descarga recursos de NLTK
    """
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN DE NLTK")
    print("=" * 60)

    try:
        import nltk

        recursos_nltk = [
            'stopwords',
            'punkt',
            'vader_lexicon',
            'averaged_perceptron_tagger',
            'wordnet'
        ]

        for recurso in recursos_nltk:
            try:
                print(f"Descargando {recurso}...")
                nltk.download(recurso, quiet=True)
                print(f"✓ {recurso} descargado")
            except Exception as e:
                print(f"⚠ Error descargando {recurso}: {e}")

        print("✓ Configuración de NLTK completada")
        return True

    except ImportError:
        print("✗ NLTK no está instalado")
        return False
    except Exception as e:
        print(f"✗ Error configurando NLTK: {e}")
        return False


def configurar_spacy():
    """
    Configura spaCy y descarga el modelo en español
    """
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN DE SPACY")
    print("=" * 60)

    try:
        import spacy

        # Verificar si el modelo ya está instalado
        try:
            nlp = spacy.load("es_core_news_sm")
            print("✓ Modelo es_core_news_sm ya está instalado")
            return True
        except OSError:
            print("⚠ Modelo es_core_news_sm no encontrado, descargando...")

            try:
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", "es_core_news_sm"
                ])
                print("✓ Modelo es_core_news_sm instalado correctamente")
                return True
            except subprocess.CalledProcessError as e:
                print(f"✗ Error descargando modelo spaCy: {e}")
                print("Intenta manualmente: python -m spacy download es_core_news_sm")
                return False

    except ImportError:
        print("✗ spaCy no está instalado")
        return False


def crear_archivos_ejemplo():
    """
    Crea archivos de ejemplo para probar el analizador
    """
    print("\n" + "=" * 60)
    print("CREANDO ARCHIVOS DE EJEMPLO")
    print("=" * 60)

    # Crear carpeta de ejemplos
    carpeta_ejemplos = Path("textos_ejemplo")
    carpeta_ejemplos.mkdir(exist_ok=True)
    print(f"✓ Carpeta creada: {carpeta_ejemplos}")

    ejemplos = {
        "inteligencia_artificial.txt": """
La inteligencia artificial representa una de las revoluciones tecnológicas más significativas de nuestro tiempo. 
Esta disciplina, que combina algoritmos avanzados con grandes volúmenes de datos, está transformando 
prácticamente todos los sectores de la economía global.

Los sistemas de machine learning han demostrado capacidades extraordinarias en tareas que tradicionalmente 
requerían inteligencia humana. Desde el reconocimiento de imágenes hasta el procesamiento de lenguaje natural, 
estas tecnologías están superando las expectativas más optimistas de los investigadores.

Sin embargo, el desarrollo de la IA también plantea desafíos éticos y sociales importantes. La automatización 
podría desplazar empleos tradicionales, mientras que los algoritmos de decisión pueden perpetuar sesgos 
existentes en los datos de entrenamiento.

Es fundamental que la sociedad aborde estos retos de manera proactiva, estableciendo marcos regulatorios 
adecuados y promoviendo el desarrollo responsable de estas tecnologías. Solo así podremos maximizar los 
beneficios de la IA mientras minimizamos sus riesgos potenciales.

El futuro de la inteligencia artificial dependerá de nuestra capacidad para equilibrar la innovación 
tecnológica con la responsabilidad social, asegurando que estos avances sirvan al bienestar de toda la humanidad.
        """,

        "cambio_climatico.txt": """
El cambio climático constituye uno de los desafíos más apremiantes de la actualidad. Los datos científicos 
muestran de manera inequívoca que las actividades humanas están alterando el sistema climático global 
a un ritmo sin precedentes en la historia de la humanidad.

Las emisiones de gases de efecto invernadero, principalmente dióxido de carbono, han aumentado 
dramáticamente desde la revolución industrial. Este incremento está provocando el calentamiento global, 
con consecuencias que ya son visibles en todo el planeta.

Los efectos del cambio climático incluyen el aumento del nivel del mar, la intensificación de fenómenos 
meteorológicos extremos, cambios en los patrones de precipitación y la alteración de ecosistemas completos. 
Estas transformaciones amenazan la seguridad alimentaria, la disponibilidad de agua dulce y la habitabilidad 
de muchas regiones del mundo.

La transición hacia energías renovables es fundamental para mitigar estos efectos. Las tecnologías solares, 
eólicas e hidroeléctricas han alcanzado niveles de eficiencia y costos que las hacen competitivas con 
los combustibles fósiles tradicionales.

Además de la mitigación, es crucial desarrollar estrategias de adaptación que permitan a las sociedades 
ajustarse a los cambios inevitables. Esto incluye la construcción de infraestructuras resilientes, 
la protección de ecosistemas y la implementación de sistemas de alerta temprana.
        """,

        "educacion_digital.txt": """
La educación digital ha experimentado una transformación acelerada, especialmente tras los desafíos 
globales que obligaron a instituciones educativas de todo el mundo a adoptar modalidades virtuales 
de enseñanza y aprendizaje.

Las plataformas de aprendizaje en línea han democratizado el acceso a la educación, permitiendo que 
estudiantes de diversas ubicaciones geográficas y contextos socioeconómicos accedan a contenidos educativos 
de alta calidad. Esta democratización representa una oportunidad sin precedentes para reducir las 
brechas educativas globales.

Sin embargo, la implementación efectiva de la educación digital requiere más que simplemente trasladar 
contenidos tradicionales a formatos digitales. Es necesario repensar las metodologías pedagógicas, 
desarrollar nuevas competencias digitales tanto en docentes como en estudiantes, y garantizar el acceso 
equitativo a las tecnologías necesarias.

Los datos y la analítica educativa están proporcionando insights valiosos sobre los procesos de aprendizaje, 
permitiendo personalizar la experiencia educativa de manera más efectiva. Los sistemas adaptativos pueden 
ajustar el ritmo y el contenido según las necesidades individuales de cada estudiante.

La gamificación y las simulaciones interactivas están haciendo que el aprendizaje sea más atractivo y 
efectivo, especialmente para las nuevas generaciones que han crecido inmersas en entornos digitales. 
Estas herramientas pueden transformar conceptos abstractos en experiencias tangibles y memorables.
        """
    }

    archivos_creados = 0
    for nombre_archivo, contenido in ejemplos.items():
        ruta_archivo = carpeta_ejemplos / nombre_archivo
        try:
            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                f.write(contenido.strip())
            print(f"✓ Creado: {nombre_archivo}")
            archivos_creados += 1
        except Exception as e:
            print(f"✗ Error creando {nombre_archivo}: {e}")

    print(f"\n✓ {archivos_creados} archivos de ejemplo creados en '{carpeta_ejemplos}'")
    return archivos_creados > 0


def verificar_instalacion():
    """
    Verifica que la instalación sea correcta
    """
    print("\n" + "=" * 60)
    print("VERIFICACIÓN FINAL DE LA INSTALACIÓN")
    print("=" * 60)

    # Verificar archivo principal
    if not Path("analizador_texto_avanzado_corregido.py").exists():
        print("✗ Archivo principal no encontrado: analizador_texto_avanzado_corregido.py")
        return False

    try:
        # Importar y probar
        sys.path.insert(0, '.')
        from analizador_texto_avanzado_corregido import AnalizadorTextoAvanzado

        analizador = AnalizadorTextoAvanzado()
        print("✓ Analizador importado correctamente")

        # Prueba básica
        texto_prueba = "Este es un texto de prueba para verificar la funcionalidad básica."
        resultados = analizador.analizar_texto_completo(texto_prueba)

        if resultados and 'resumen_ejecutivo' in resultados:
            print("✓ Análisis básico funciona correctamente")
            return True
        else:
            print("✗ El análisis no generó resultados esperados")
            return False

    except ImportError as e:
        print(f"✗ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"✗ Error en verificación: {e}")
        return False


def mostrar_instrucciones_uso():
    """
    Muestra instrucciones de uso
    """
    print("\n" + "=" * 60)
    print("INSTRUCCIONES DE USO")
    print("=" * 60)

    print("""
🎉 ¡Instalación completada exitosamente!

FORMAS DE USAR EL ANALIZADOR:

1. Ejemplo básico:
   python ejemplo_uso_corregido.py

2. Análisis de archivo:
   python analizador_texto_avanzado_corregido.py --archivo mi_texto.txt

3. Análisis de carpeta:
   python analizador_texto_avanzado_corregido.py --carpeta textos_ejemplo/

4. Modo interactivo:
   python analizador_texto_avanzado_corregido.py

5. Análisis comparativo:
   python analizador_texto_avanzado_corregido.py --carpeta textos_ejemplo/ --comparar

ARCHIVOS DISPONIBLES:
- analizador_texto_avanzado_corregido.py (módulo principal)
- ejemplo_uso_corregido.py (ejemplo de uso)
- textos_ejemplo/ (carpeta con archivos de prueba)

PRÓXIMOS PASOS:
1. Ejecuta: python ejemplo_uso_corregido.py
2. Experimenta con tus propios textos
3. Revisa los reportes HTML generados

¡Disfruta analizando textos! 📊📈
    """)


def main():
    """
    Función principal de configuración
    """
    print("=" * 60)
    print("CONFIGURADOR DEL ANALIZADOR DE TEXTO AVANZADO")
    print("=" * 60)
    print("Este script configurará automáticamente el entorno completo\n")

    # Verificar Python
    if sys.version_info < (3, 7):
        print("❌ Se requiere Python 3.7 o superior")
        print(f"Versión actual: {sys.version}")
        return False

    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detectado")

    # Paso 1: Instalar dependencias
    if not verificar_e_instalar_dependencias():
        print("\n❌ Falló la instalación de dependencias críticas")
        return False

    # Paso 2: Configurar NLTK
    if not configurar_nltk():
        print("\n⚠ Advertencia: NLTK no se configuró completamente")

    # Paso 3: Configurar spaCy
    if not configurar_spacy():
        print("\n⚠ Advertencia: spaCy no se configuró completamente")
        print("El analizador funcionará con funcionalidad limitada")

    # Paso 4: Crear archivos de ejemplo
    crear_archivos_ejemplo()

    # Paso 5: Verificación final
    if verificar_instalacion():
        mostrar_instrucciones_uso()
        return True
    else:
        print("\n❌ La verificación final falló")
        print("Revisa los errores anteriores e intenta nuevamente")
        return False


if __name__ == "__main__":
    try:
        exito = main()
        if exito:
            print("\n🎉 ¡Configuración completada exitosamente!")
        else:
            print("\n❌ La configuración no se completó correctamente")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nConfiguración cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError inesperado durante la configuración: {e}")
        sys.exit(1)