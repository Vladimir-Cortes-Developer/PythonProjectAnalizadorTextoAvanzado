# prueba_rapida.py
# Script independiente para verificar que todo funciona

import os
import sys


def verificar_archivos():
    """Verifica que todos los archivos necesarios existan"""
    print("Verificando archivos del proyecto...")

    archivos_requeridos = [
        'analizador_texto_avanzado.py',
        'ejemplo_uso.py',
        'analizar_archivo.py',
        'setup.py'
    ]

    archivos_faltantes = []

    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            print(f"✓ {archivo}")
        else:
            print(f"✗ {archivo} - FALTA")
            archivos_faltantes.append(archivo)

    if archivos_faltantes:
        print(f"\nERROR: Faltan archivos importantes:")
        for archivo in archivos_faltantes:
            print(f"  - {archivo}")
        return False

    return True


def verificar_dependencias():
    """Verifica que las dependencias estén instaladas"""
    print("\nVerificando dependencias...")

    dependencias = [
        ('nltk', 'Natural Language Toolkit'),
        ('pandas', 'Data Analysis Library'),
        ('numpy', 'Numerical Computing'),
        ('matplotlib', 'Plotting Library'),
        ('seaborn', 'Statistical Visualization'),
        ('wordcloud', 'Word Cloud Generator'),
        ('plotly', 'Interactive Plotting'),
        ('spacy', 'Advanced NLP'),
        ('textblob', 'Text Processing'),
    ]

    dependencias_faltantes = []

    for modulo, descripcion in dependencias:
        try:
            __import__(modulo)
            print(f"✓ {modulo} - {descripcion}")
        except ImportError:
            print(f"✗ {modulo} - NO INSTALADO - {descripcion}")
            dependencias_faltantes.append(modulo)

    # Verificar spaCy específicamente
    try:
        import spacy
        nlp = spacy.load("es_core_news_sm")
        print("✓ spacy modelo es_core_news_sm - Disponible")
    except (ImportError, OSError):
        print("⚠ spacy modelo es_core_news_sm - NO DISPONIBLE (funcionalidad limitada)")
        print("  Instala con: python -m spacy download es_core_news_sm")

    if dependencias_faltantes:
        print(f"\nAdvertencia: Dependencias faltantes:")
        for dep in dependencias_faltantes:
            print(f"  - {dep}")
        print(f"\nPuedes instalarlas con:")
        print(f"pip install {' '.join(dependencias_faltantes)}")
        return False

    return True


def probar_analizador():
    """Prueba el funcionamiento básico del analizador"""
    print("\nProbando el analizador...")

    try:
        # Intentar importar
        from analizador_texto_avanzado import AnalizadorTextoAvanzado
        print("✓ Importación exitosa")

        # Crear instancia
        analizador = AnalizadorTextoAvanzado()
        print("✓ Instancia creada")

        # Texto de prueba
        texto_prueba = """
        La inteligencia artificial está revolucionando el mundo moderno.
        Los algoritmos de machine learning procesan grandes cantidades de datos
        para encontrar patrones ocultos y realizar predicciones precisas.
        Esta tecnología tiene aplicaciones en medicina, finanzas, educación
        y muchos otros campos importantes para la sociedad.
        """

        # Realizar análisis
        print("Ejecutando análisis completo...")
        resultados = analizador.analizar_texto_completo(texto_prueba)
        print("✓ Análisis completado")

        # Verificar resultados
        resumen = resultados.get('resumen_ejecutivo', {})

        print(f"\nResultados del análisis:")
        print(f"  - Palabras totales: {resumen.get('total_palabras', 'N/A')}")
        print(f"  - Palabras únicas: {resumen.get('palabras_unicas', 'N/A')}")
        print(f"  - Diversidad léxica: {resumen.get('diversidad_lexica', 'N/A')}")
        print(f"  - Sentimiento: {resumen.get('sentimiento_general', 'N/A')}")
        print(f"  - Complejidad: {resumen.get('complejidad', 'N/A')}")

        # Verificar que haya palabras frecuentes
        if 'frecuencias' in resultados and 'mas_frecuentes' in resultados['frecuencias']:
            palabras_freq = resultados['frecuencias']['mas_frecuentes'][:5]
            print(f"  - Top 5 palabras: {[p[0] for p in palabras_freq]}")

        print("✓ Análisis verificado correctamente")
        return True

    except ImportError as e:
        print(f"✗ Error de importación: {e}")
        print("Asegúrate de que el archivo 'analizador_texto_avanzado.py' esté presente")
        return False
    except Exception as e:
        print(f"✗ Error durante la prueba: {e}")
        return False


def probar_archivos_ejemplo():
    """Prueba los archivos de ejemplo"""
    print("\nVerificando archivos de ejemplo...")

    carpeta_ejemplos = 'textos_ejemplo'

    if not os.path.exists(carpeta_ejemplos):
        print(f"⚠ Carpeta {carpeta_ejemplos}/ no existe")
        print("Ejecuta 'python setup.py' para crearla")
        return False

    archivos_ejemplo = [
        'inteligencia_artificial.txt',
        'cambio_climatico.txt',
        'educacion_digital.txt'
    ]

    archivos_encontrados = []

    for archivo in archivos_ejemplo:
        ruta_archivo = os.path.join(carpeta_ejemplos, archivo)
        if os.path.exists(ruta_archivo):
            try:
                with open(ruta_archivo, 'r', encoding='utf-8') as f:
                    contenido = f.read()
                print(f"✓ {archivo} ({len(contenido)} caracteres)")
                archivos_encontrados.append(ruta_archivo)
            except Exception as e:
                print(f"✗ Error leyendo {archivo}: {e}")
        else:
            print(f"⚠ {archivo} - No encontrado")

    if archivos_encontrados:
        print(f"✓ {len(archivos_encontrados)} archivos de ejemplo disponibles")
        return True
    else:
        print("⚠ No hay archivos de ejemplo disponibles")
        return False


def probar_generacion_reportes():
    """Prueba la generación de reportes"""
    print("\nProbando generación de reportes...")

    try:
        from analizador_texto_avanzado import AnalizadorTextoAvanzado

        analizador = AnalizadorTextoAvanzado()

        # Texto simple para reporte
        texto_simple = """
        Este es un texto de prueba para verificar la generación de reportes.
        El analizador debe ser capaz de procesar este contenido y generar
        diferentes tipos de archivos de salida incluyendo HTML, JSON y TXT.
        """

        # Análisis
        resultados = analizador.analizar_texto_completo(texto_simple)

        # Intentar generar un reporte JSON (más simple)
        try:
            archivo_json = analizador.generar_reporte_completo(formato='json')
            if os.path.exists(archivo_json):
                print(f"✓ Reporte JSON generado: {archivo_json}")
                # Limpiar archivo de prueba
                os.remove(archivo_json)
                return True
            else:
                print("✗ Reporte JSON no se generó correctamente")
                return False
        except Exception as e:
            print(f"✗ Error generando reporte: {e}")
            return False

    except Exception as e:
        print(f"✗ Error en prueba de reportes: {e}")
        return False


def mostrar_resumen_final(resultados):
    """Muestra un resumen final de las pruebas"""
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS")
    print("=" * 60)

    total_pruebas = len(resultados)
    pruebas_exitosas = sum(resultados.values())
    porcentaje = (pruebas_exitosas / total_pruebas) * 100

    for nombre, resultado in resultados.items():
        estado = "✓ PASÓ" if resultado else "✗ FALLÓ"
        print(f"{estado} - {nombre}")

    print(f"\nResultado: {pruebas_exitosas}/{total_pruebas} pruebas exitosas ({porcentaje:.1f}%)")

    if porcentaje == 100:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("El analizador está listo para usar")

        print("\nPróximos pasos:")
        print("1. Ejecuta: python ejemplo_uso.py")
        print("2. O ejecuta: python analizar_archivo.py")
        print("3. Para uso avanzado: python analizador_texto_avanzado.py --help")

    elif porcentaje >= 75:
        print("\n⚠ PRUEBAS MAYORMENTE EXITOSAS")
        print("El analizador debería funcionar con funcionalidad limitada")
        print("Revisa las pruebas que fallaron e instala dependencias faltantes")

    else:
        print("\n❌ VARIAS PRUEBAS FALLARON")
        print("Es necesario resolver los problemas antes de usar el analizador")
        print("\nSoluciones recomendadas:")
        print("1. Ejecuta: python setup.py")
        print("2. Instala dependencias: pip install nltk pandas matplotlib seaborn wordcloud plotly spacy textblob")
        print("3. Verifica que todos los archivos estén presentes")


def main():
    """Función principal de pruebas"""
    print("=" * 60)
    print("PRUEBA RÁPIDA DEL ANALIZADOR DE TEXTO AVANZADO")
    print("=" * 60)
    print("Esta prueba verificará que todo esté configurado correctamente\n")

    # Ejecutar todas las pruebas
    resultados_pruebas = {}

    # Prueba 1: Archivos
    resultados_pruebas["Archivos del proyecto"] = verificar_archivos()

    # Prueba 2: Dependencias
    resultados_pruebas["Dependencias de Python"] = verificar_dependencias()

    # Prueba 3: Funcionalidad básica
    resultados_pruebas["Funcionalidad del analizador"] = probar_analizador()

    # Prueba 4: Archivos de ejemplo
    resultados_pruebas["Archivos de ejemplo"] = probar_archivos_ejemplo()

    # Prueba 5: Generación de reportes
    resultados_pruebas["Generación de reportes"] = probar_generacion_reportes()

    # Mostrar resumen
    mostrar_resumen_final(resultados_pruebas)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrueba cancelada por el usuario")
    except Exception as e:
        print(f"\n\nError inesperado durante las pruebas: {e}")
        print("Por favor, reporta este error si persiste")