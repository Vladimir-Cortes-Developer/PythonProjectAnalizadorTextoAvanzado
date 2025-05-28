# setup.py - Script de configuración automática completo
import os
import subprocess
import sys
import platform
import ssl


def configurar_ssl():
    """Configura SSL para evitar errores de certificado"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


def mostrar_bienvenida():
    """Muestra mensaje de bienvenida"""
    print("=" * 60)
    print("CONFIGURACIÓN DEL ANALIZADOR DE TEXTO AVANZADO")
    print("=" * 60)
    print("Este script instalará todas las dependencias necesarias")
    print("y configurará el entorno para el analizador de texto.")
    print("=" * 60)


def verificar_python():
    """Verifica la versión de Python"""
    version = sys.version_info
    print(f"Versión de Python detectada: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("ERROR: Se requiere Python 3.7 o superior")
        return False

    print("✓ Versión de Python compatible")
    return True


def crear_estructura_carpetas():
    """Crea la estructura de carpetas necesaria"""
    print("\nCreando estructura de carpetas...")

    carpetas = [
        'resultados',
        'textos_ejemplo',
        'reportes',
        'visualizaciones',
        'datos_temp'
    ]

    for carpeta in carpetas:
        try:
            if not os.path.exists(carpeta):
                os.makedirs(carpeta)
                print(f"✓ Carpeta creada: {carpeta}/")
            else:
                print(f"○ Carpeta ya existe: {carpeta}/")
        except Exception as e:
            print(f"✗ Error creando {carpeta}: {e}")


def instalar_dependencias():
    """Instala todas las dependencias necesarias"""
    print("\nInstalando dependencias de Python...")

    dependencias = [
        'nltk>=3.8',
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'wordcloud>=1.9.0',
        'plotly>=5.10.0',
        'spacy>=3.4.0',
        'textblob>=0.17.0',
        'vaderSentiment>=3.3.0'
    ]

    errores = []

    for dep in dependencias:
        try:
            print(f"Instalando {dep}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', dep, '--quiet'
            ])
            print(f"✓ {dep} instalado correctamente")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error instalando {dep}")
            errores.append(dep)

    if errores:
        print(f"\nAdvertencia: No se pudieron instalar: {', '.join(errores)}")
        print("Intenta instalarlas manualmente con:")
        for dep in errores:
            print(f"pip install {dep}")

    return len(errores) == 0


def configurar_nltk():
    """Configura y descarga recursos de NLTK"""
    print("\nConfigurando NLTK...")

    configurar_ssl()

    try:
        import nltk

        # Crear directorio de datos de NLTK si no existe
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)

        # Lista completa de recursos necesarios (incluyendo punkt_tab)
        recursos_nltk = [
            'punkt',
            'punkt_tab',  # Nuevo recurso requerido
            'stopwords',
            'vader_lexicon',
            'averaged_perceptron_tagger',
            'wordnet',
            'omw-1.4'
        ]

        for recurso in recursos_nltk:
            try:
                print(f"Descargando {recurso}...")
                nltk.download(recurso, quiet=False)
                print(f"✓ {recurso} descargado")
            except Exception as e:
                print(f"⚠ Advertencia descargando {recurso}: {e}")

                # Intento alternativo para punkt_tab
                if recurso == 'punkt_tab':
                    try:
                        print("Intentando descarga alternativa...")
                        nltk.download('punkt_tab', download_dir=nltk_data_dir)
                        print("✓ punkt_tab descargado (alternativo)")
                    except:
                        print("⚠ punkt_tab no disponible, se usará fallback")

        print("✓ Configuración de NLTK completa")
        return True

    except ImportError:
        print("✗ NLTK no está instalado correctamente")
        return False


def configurar_spacy():
    """Configura spaCy y descarga el modelo en español"""
    print("\nConfigurando spaCy...")

    try:
        # Verificar si spaCy está instalado
        import spacy

        # Intentar descargar el modelo en español
        print("Descargando modelo en español para spaCy...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'spacy', 'download', 'es_core_news_sm'
            ])
            print("✓ Modelo es_core_news_sm descargado")

            # Verificar que el modelo se cargue correctamente
            nlp = spacy.load("es_core_news_sm")
            print("✓ Modelo verificado correctamente")
            return True

        except subprocess.CalledProcessError:
            print("⚠ Error descargando modelo de spaCy")
            print("Puedes instalarlo manualmente con:")
            print("python -m spacy download es_core_news_sm")
            return False

    except ImportError:
        print("✗ spaCy no está instalado correctamente")
        return False


def crear_archivos_ejemplo():
    """Crea archivos de ejemplo para pruebas"""
    print("\nCreando archivos de ejemplo...")

    # Texto de ejemplo sobre IA
    texto_ia = """La inteligencia artificial representa una de las revoluciones tecnológicas más importantes de nuestro tiempo. 
Esta disciplina combina algoritmos sofisticados, grandes volúmenes de datos y poder computacional para crear 
sistemas que pueden realizar tareas que tradicionalmente requerían inteligencia humana.

El machine learning, una rama fundamental de la IA, permite a las máquinas aprender patrones de los datos 
sin ser programadas explícitamente para cada tarea específica. Esto ha llevado a avances extraordinarios 
en campos como el reconocimiento de imágenes, procesamiento de lenguaje natural, y sistemas de recomendación.

Sin embargo, el desarrollo de la IA también plantea desafíos éticos y sociales importantes. Cuestiones como 
la privacidad de datos, el sesgo algorítmico, y el impacto en el empleo requieren una consideración cuidadosa 
mientras avanzamos hacia un futuro más automatizado.

La colaboración entre humanos y máquinas, más que el reemplazo completo, parece ser el camino más prometedor 
para aprovechar al máximo el potencial de la inteligencia artificial mientras preservamos los valores humanos fundamentales."""

    # Texto de ejemplo sobre cambio climático
    texto_clima = """El cambio climático es uno de los desafíos más urgentes que enfrenta la humanidad en el siglo XXI. 
Las evidencias científicas muestran que las actividades humanas, particularmente la emisión de gases 
de efecto invernadero, están alterando el sistema climático global de manera significativa.

Los impactos del cambio climático ya son visibles en todo el mundo: aumento de las temperaturas globales, 
derretimiento de glaciares, elevación del nivel del mar, y cambios en los patrones de precipitación. 
Estos cambios tienen consecuencias profundas para los ecosistemas, la agricultura, y las comunidades humanas.

La transición hacia energías renovables y tecnologías limpias es fundamental para mitigar estos efectos. 
Sin embargo, la adaptación también es crucial, ya que algunos cambios climáticos son inevitables debido 
a las emisiones pasadas y presentes.

La cooperación internacional, la innovación tecnológica, y los cambios en los estilos de vida son 
elementos clave para abordar este desafío global de manera efectiva."""

    # Texto de ejemplo sobre educación
    texto_educacion = """La educación es la base fundamental para el desarrollo personal y social de los individuos. 
En la era digital, los métodos de enseñanza y aprendizaje están experimentando una transformación 
significativa, incorporando nuevas tecnologías y enfoques pedagógicos innovadores.

Las plataformas de aprendizaje en línea, la realidad virtual, y la inteligencia artificial están 
revolucionando la forma en que accedemos al conocimiento. Estas herramientas permiten personalizar 
la experiencia educativa, adaptándose a las necesidades y ritmos de aprendizaje individuales.

Sin embargo, la tecnología no puede reemplazar completamente la interacción humana en el proceso educativo. 
Los educadores siguen siendo fundamentales para guiar, motivar, y proporcionar el contexto social 
necesario para un aprendizaje efectivo.

El futuro de la educación probablemente combinará lo mejor de ambos mundos: la eficiencia y personalización 
de la tecnología con la sabiduría y empatía humana."""

    ejemplos = [
        ('inteligencia_artificial.txt', texto_ia),
        ('cambio_climatico.txt', texto_clima),
        ('educacion_digital.txt', texto_educacion)
    ]

    for nombre_archivo, contenido in ejemplos:
        ruta_archivo = os.path.join('textos_ejemplo', nombre_archivo)
        try:
            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                f.write(contenido.strip())
            print(f"✓ Archivo creado: {ruta_archivo}")
        except Exception as e:
            print(f"✗ Error creando {ruta_archivo}: {e}")


def crear_script_prueba():
    """Crea un script de prueba rápida"""
    print("\nCreando script de prueba...")

    script_prueba = '''# prueba_rapida.py
# Script para probar que el analizador funciona correctamente

from analizador_texto_avanzado import AnalizadorTextoAvanzado
import os

def main():
    print("PRUEBA RÁPIDA DEL ANALIZADOR DE TEXTO")
    print("=" * 40)

    # Verificar que los archivos principales existen
    archivos_requeridos = [
        'analizador_texto_avanzado.py',
        'ejemplo_uso.py',
        'analizar_archivo.py'
    ]

    print("Verificando archivos...")
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            print(f"✓ {archivo}")
        else:
            print(f"✗ {archivo} - FALTA")
            return False

    # Probar importación
    try:
        print("\\nProbando importación...")
        analizador = AnalizadorTextoAvanzado()
        print("✓ Analizador importado correctamente")
    except Exception as e:
        print(f"✗ Error importando: {e}")
        return False

    # Probar análisis básico
    try:
        print("\\nProbando análisis básico...")
        texto_prueba = """
        Este es un texto de prueba para verificar que el analizador 
        funciona correctamente. La inteligencia artificial está 
        transformando nuestro mundo de manera extraordinaria.
        """

        resultados = analizador.analizar_texto_completo(texto_prueba)

        print("✓ Análisis completado")
        print(f"  - Palabras analizadas: {resultados['resumen_ejecutivo']['total_palabras']}")
        print(f"  - Diversidad léxica: {resultados['resumen_ejecutivo']['diversidad_lexica']}")
        print(f"  - Sentimiento: {resultados['resumen_ejecutivo']['sentimiento_general']}")

    except Exception as e:
        print(f"✗ Error en análisis: {e}")
        return False

    # Probar archivos de ejemplo
    try:
        print("\\nProbando archivos de ejemplo...")
        archivo_ejemplo = os.path.join('textos_ejemplo', 'inteligencia_artificial.txt')
        if os.path.exists(archivo_ejemplo):
            with open(archivo_ejemplo, 'r', encoding='utf-8') as f:
                contenido = f.read()
            print(f"✓ Archivo de ejemplo leído ({len(contenido)} caracteres)")
        else:
            print("⚠ Archivo de ejemplo no encontrado")
    except Exception as e:
        print(f"✗ Error leyendo ejemplo: {e}")

    print("\\n" + "=" * 40)
    print("¡PRUEBA COMPLETADA EXITOSAMENTE!")
    print("=" * 40)
    print("\\nPróximos pasos:")
    print("1. Ejecuta: python ejemplo_uso.py")
    print("2. O ejecuta: python analizar_archivo.py")
    print("3. Los reportes se guardarán automáticamente")

    return True

if __name__ == "__main__":
    main()
'''

    try:
        with open('prueba_rapida.py', 'w', encoding='utf-8') as f:
            f.write(script_prueba)
        print("✓ Script de prueba creado: prueba_rapida.py")
    except Exception as e:
        print(f"✗ Error creando script de prueba: {e}")


def crear_fix_nltk():
    """Crea el script para arreglar problemas de NLTK"""
    print("\nCreando script de reparación NLTK...")

    script_fix = '''# fix_nltk.py - Script para arreglar problemas de NLTK
import nltk
import ssl

def configurar_ssl():
    """Configura SSL para evitar errores de certificado"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

def descargar_recursos_nltk():
    """Descarga todos los recursos necesarios de NLTK"""
    configurar_ssl()

    print("Descargando recursos de NLTK...")

    # Lista completa de recursos necesarios (incluyendo los nuevos)
    recursos = [
        'punkt',
        'punkt_tab',  # Nuevo recurso requerido
        'stopwords',
        'vader_lexicon',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4'
    ]

    errores = []

    for recurso in recursos:
        try:
            print(f"Descargando {recurso}...")
            nltk.download(recurso, quiet=False)
            print(f"✓ {recurso} descargado exitosamente")
        except Exception as e:
            print(f"✗ Error descargando {recurso}: {e}")
            errores.append(recurso)

    if errores:
        print(f"\\nRecursos que fallaron: {errores}")
        print("Intentando descarga completa...")
        try:
            nltk.download('all')
            print("✓ Descarga completa exitosa")
        except Exception as e:
            print(f"✗ Error en descarga completa: {e}")

    # Verificar que los recursos están disponibles
    verificar_recursos()

def verificar_recursos():
    """Verifica que los recursos estén correctamente instalados"""
    print("\\nVerificando recursos...")

    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords

        # Probar tokenización
        texto_prueba = "Hola mundo. Este es un texto de prueba."

        # Probar sent_tokenize (usa punkt_tab)
        oraciones = sent_tokenize(texto_prueba, language='spanish')
        print(f"✓ Tokenización de oraciones funciona: {len(oraciones)} oraciones")

        # Probar word_tokenize
        palabras = word_tokenize(texto_prueba, language='spanish')
        print(f"✓ Tokenización de palabras funciona: {len(palabras)} palabras")

        # Probar stopwords
        stop_words = stopwords.words('spanish')
        print(f"✓ Stopwords españolas disponibles: {len(stop_words)} palabras")

        print("\\n✅ Todos los recursos funcionan correctamente!")
        return True

    except Exception as e:
        print(f"\\nError verificando recursos: {e}")
        return False

if __name__ == "__main__":
    print("SOLUCIONADOR DE PROBLEMAS NLTK")
    print("=" * 40)
    descargar_recursos_nltk()
'''

    try:
        with open('fix_nltk.py', 'w', encoding='utf-8') as f:
            f.write(script_fix)
        print("✓ Script de reparación creado: fix_nltk.py")
    except Exception as e:
        print(f"✗ Error creando script de reparación: {e}")


def verificar_instalacion():
    """Verifica que todo esté correctamente instalado"""
    print("\nVerificando instalación completa...")

    verificaciones = []

    # Verificar librerías principales
    librerias = [
        'nltk', 'pandas', 'numpy', 'matplotlib',
        'seaborn', 'wordcloud', 'plotly', 'spacy', 'textblob'
    ]

    print("Verificando librerías...")
    for lib in librerias:
        try:
            __import__(lib)
            print(f"✓ {lib}")
            verificaciones.append(True)
        except ImportError:
            print(f"✗ {lib} - NO INSTALADA")
            verificaciones.append(False)

    # Verificar modelo de spaCy
    try:
        import spacy
        nlp = spacy.load("es_core_news_sm")
        print("✓ Modelo spaCy es_core_news_sm")
        verificaciones.append(True)
    except (ImportError, OSError):
        print("⚠ Modelo spaCy es_core_news_sm - NO DISPONIBLE")
        verificaciones.append(False)

    # Verificar archivos
    archivos_clave = [
        'analizador_texto_avanzado.py',
        'ejemplo_uso.py',
        'analizar_archivo.py',
        'prueba_rapida.py',
        'fix_nltk.py'
    ]

    print("\\nVerificando archivos...")
    for archivo in archivos_clave:
        if os.path.exists(archivo):
            print(f"✓ {archivo}")
            verificaciones.append(True)
        else:
            print(f"✗ {archivo} - FALTA")
            verificaciones.append(False)

    # Verificar carpetas
    carpetas = ['resultados', 'textos_ejemplo', 'reportes', 'visualizaciones']
    print("\\nVerificando carpetas...")
    for carpeta in carpetas:
        if os.path.exists(carpeta):
            print(f"✓ {carpeta}/")
            verificaciones.append(True)
        else:
            print(f"✗ {carpeta}/ - FALTA")
            verificaciones.append(False)

    # Resultado final
    exitos = sum(verificaciones)
    total = len(verificaciones)
    porcentaje = (exitos / total) * 100

    print(f"\\nRESULTADO: {exitos}/{total} verificaciones exitosas ({porcentaje:.1f}%)")

    if porcentaje >= 80:
        print("✓ INSTALACIÓN EXITOSA - El analizador está listo para usar")
        return True
    else:
        print("✗ INSTALACIÓN INCOMPLETA - Revisa los errores anteriores")
        return False


def mostrar_instrucciones_uso():
    """Muestra instrucciones de uso"""
    print("\\n" + "=" * 60)
    print("INSTRUCCIONES DE USO")
    print("=" * 60)

    print("\\n1. ARREGLAR PROBLEMAS NLTK (si es necesario):")
    print("   python fix_nltk.py")

    print("\\n2. PRUEBA RÁPIDA:")
    print("   python prueba_rapida.py")

    print("\\n3. ANÁLISIS BÁSICO:")
    print("   python ejemplo_uso.py")

    print("\\n4. ANÁLISIS INTERACTIVO:")
    print("   python analizar_archivo.py")

    print("\\n5. LÍNEA DE COMANDOS:")
    print("   python analizador_texto_avanzado.py --archivo mi_texto.txt")
    print("   python analizador_texto_avanzado.py --carpeta mis_textos/")
    print("   python analizador_texto_avanzado.py --carpeta mis_textos/ --comparar")

    print("\\n6. ARCHIVOS DE EJEMPLO:")
    print("   Los archivos de ejemplo están en la carpeta 'textos_ejemplo/'")
    print("   Puedes usarlos para probar el analizador")

    print("\\n7. RESULTADOS:")
    print("   Los reportes se guardan automáticamente en:")
    print("   - Archivos HTML (visuales)")
    print("   - Archivos JSON (datos)")
    print("   - Archivos TXT (texto plano)")
    print("   - Nube de palabras (PNG)")
    print("   - Dashboard interactivo (HTML)")


def limpiar_instalacion():
    """Limpia archivos temporales de instalación"""
    print("\\nLimpiando archivos temporales...")

    archivos_temp = [
        '__pycache__',
        '*.pyc',
        '.pytest_cache'
    ]

    import glob
    import shutil

    for patron in archivos_temp:
        for archivo in glob.glob(patron, recursive=True):
            try:
                if os.path.isdir(archivo):
                    shutil.rmtree(archivo)
                else:
                    os.remove(archivo)
                print(f"✓ Eliminado: {archivo}")
            except Exception as e:
                print(f"✗ Error eliminando {archivo}: {e}")


def main():
    """Función principal de configuración"""
    mostrar_bienvenida()

    # Verificar Python
    if not verificar_python():
        print("\\nERROR: Versión de Python no compatible")
        return False

    # Crear estructura
    crear_estructura_carpetas()

    # Preguntar sobre instalación de dependencias
    print("\\n" + "=" * 60)
    respuesta = input("¿Deseas instalar las dependencias de Python? (s/n): ").lower()

    if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
        if not instalar_dependencias():
            print("\\nAdvertencia: Algunas dependencias no se instalaron correctamente")

        # Configurar NLTK
        if not configurar_nltk():
            print("\\nAdvertencia: Error configurando NLTK")

        # Configurar spaCy
        if not configurar_spacy():
            print("\\nAdvertencia: Error configurando spaCy (funcionalidad limitada)")
    else:
        print("\\nSaltando instalación de dependencias...")
        print("Asegúrate de instalarlas manualmente con:")
        print("pip install nltk pandas numpy matplotlib seaborn wordcloud plotly spacy textblob vaderSentiment")

    # Crear archivos de ejemplo
    crear_archivos_ejemplo()

    # Crear script de prueba
    crear_script_prueba()

    # Crear script de reparación NLTK
    crear_fix_nltk()

    # Verificar instalación
    print("\\n" + "=" * 60)
    if verificar_instalacion():
        print("\\n¡CONFIGURACIÓN COMPLETADA EXITOSAMENTE!")
        mostrar_instrucciones_uso()

        # Preguntar si ejecutar prueba
        print("\\n" + "=" * 60)
        respuesta = input("¿Deseas ejecutar la prueba rápida ahora? (s/n): ").lower()
        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            print("\\nEjecutando prueba rápida...")
            try:
                exec(open('prueba_rapida.py').read())
            except Exception as e:
                print(f"Error ejecutando prueba: {e}")
                print("Puedes ejecutarla manualmente con: python prueba_rapida.py")

        # Limpiar archivos temporales
        limpiar_instalacion()

        print("\\n✓ ¡El Analizador de Texto Avanzado está listo para usar!")
        return True
    else:
        print("\\nLa configuración no se completó correctamente")
        print("Revisa los errores anteriores e intenta nuevamente")
        print("\\nSi tienes problemas con NLTK, ejecuta: python fix_nltk.py")
        return False


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\nConfiguración cancelada por el usuario")
    except Exception as e:
        print(f"\\n\\nError inesperado durante la configuración: {e}")
        print("Por favor, reporta este error si persiste")