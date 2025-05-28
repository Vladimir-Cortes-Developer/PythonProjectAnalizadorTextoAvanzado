def main():
    """
    Ejemplo de uso del analizador de texto con manejo de errores
    """
    print("=" * 60)
    print("EJEMPLO DE USO - ANALIZADOR DE TEXTO AVANZADO")
    print("=" * 60)

    try:
        # Importar el analizador
        from analizador_texto_avanzado import AnalizadorTextoAvanzado
        print("✓ Analizador importado correctamente")
    except ImportError as e:
        print(f"✗ Error de importación: {e}")
        print("Asegúrate de que el archivo 'analizador_texto_avanzado.py' esté presente")
        return

    # Crear el analizador
    try:
        analizador = AnalizadorTextoAvanzado()
        print("✓ Analizador inicializado")
    except Exception as e:
        print(f"✗ Error inicializando analizador: {e}")
        return

    # Tu texto a analizar
    texto = """
    La inteligencia artificial está revolucionando el mundo moderno. 
    Los algoritmos de machine learning permiten a las máquinas aprender 
    de grandes volúmenes de datos sin ser programadas explícitamente. 
    Esta tecnología tiene aplicaciones en medicina, finanzas, educación 
    y muchos otros campos importantes. Sin embargo, también plantea desafíos éticos 
    importantes que debemos considerar cuidadosamente.

    El futuro de la IA depende de cómo desarrollemos estas tecnologías 
    de manera responsable. Es crucial que mantengamos un equilibrio 
    entre la innovación y la protección de los derechos humanos.

    Los sistemas de inteligencia artificial pueden procesar información
    a velocidades que superan las capacidades humanas, pero requieren
    supervisión constante para evitar sesgos y errores. La colaboración
    entre humanos y máquinas será fundamental para el progreso futuro.

    Esta tecnología revolucionaria está transformando sectores completos
    y creando nuevas oportunidades de desarrollo profesional y económico.
    """

    print(f"Texto preparado ({len(texto)} caracteres)")

    # Realizar análisis completo
    print("\nIniciando análisis completo...")
    try:
        resultados = analizador.analizar_texto_completo(texto)
        print("✓ Análisis completado exitosamente")
    except Exception as e:
        print(f"✗ Error durante el análisis: {e}")
        return

    # Verificar que tenemos resultados
    if not resultados:
        print("✗ No se generaron resultados del análisis")
        return

    # Mostrar resultados principales
    print("\n" + "=" * 50)
    print("RESULTADOS DEL ANÁLISIS")
    print("=" * 50)

    try:
        # Resumen ejecutivo
        resumen = resultados.get('resumen_ejecutivo', {})
        if resumen:
            print(f"\nRESUMEN EJECUTIVO:")
            print(f"   • Palabras totales: {resumen.get('total_palabras', 'N/A')}")
            print(f"   • Palabras únicas: {resumen.get('palabras_unicas', 'N/A')}")
            print(f"   • Diversidad léxica: {resumen.get('diversidad_lexica', 'N/A')}")
            print(f"   • Sentimiento: {resumen.get('sentimiento_general', 'N/A')}")
            print(f"   • Complejidad: {resumen.get('complejidad', 'N/A')}")

            temas = resumen.get('temas_principales', [])
            if temas:
                print(f"   • Temas principales: {', '.join(temas)}")
        else:
            print("⚠ No se encontró resumen ejecutivo")

        # Top 10 palabras más frecuentes
        frecuencias = resultados.get('frecuencias', {})
        mas_frecuentes = frecuencias.get('mas_frecuentes', [])
        if mas_frecuentes:
            print(f"\nTOP 10 PALABRAS MÁS FRECUENTES:")
            for i, (palabra, freq) in enumerate(mas_frecuentes[:10], 1):
                print(f"   {i:2d}. {palabra:<15} ({freq} veces)")
        else:
            print("⚠ No se encontraron palabras frecuentes")

        # Bigramas importantes
        bigramas = resultados.get('bigramas', [])
        if bigramas:
            print(f"\nBIGRAMAS MÁS COMUNES:")
            for i, (bigrama, freq) in enumerate(bigramas[:5], 1):
                print(f"   {i}. '{' '.join(bigrama)}' ({freq} veces)")
        else:
            print("⚠ No se encontraron bigramas")

        # Mostrar análisis de sentimientos detallado
        sentimientos = resultados.get('sentimientos', {})
        if sentimientos:
            print(f"\nANÁLISIS DE SENTIMIENTOS:")
            if 'textblob' in sentimientos:
                sent = sentimientos['textblob']
                print(f"   • TextBlob - Polaridad: {sent.get('polaridad', 'N/A')} ({sent.get('clasificacion', 'N/A')})")
                print(f"   • TextBlob - Subjetividad: {sent.get('subjetividad', 'N/A')}")

            if 'vader' in sentimientos:
                vader = sentimientos['vader']
                print(f"   • VADER - Compuesto: {vader.get('compuesto', 'N/A')} ({vader.get('clasificacion', 'N/A')})")
                print(f"   • VADER - Positivo: {vader.get('positivo', 'N/A')}")
                print(f"   • VADER - Neutral: {vader.get('neutral', 'N/A')}")
                print(f"   • VADER - Negativo: {vader.get('negativo', 'N/A')}")

            if not sentimientos.get('textblob') and not sentimientos.get('vader'):
                print("   ⚠ Análisis de sentimientos no disponible")

        # Mostrar métricas de legibilidad
        legibilidad = resultados.get('legibilidad', {})
        if legibilidad and 'error' not in legibilidad:
            print(f"\nMÉTRICAS DE LEGIBILIDAD:")
            print(f"   • Oraciones: {legibilidad.get('oraciones', 'N/A')}")
            print(f"   • Palabras por oración: {legibilidad.get('palabras_por_oracion', 'N/A')}")
            print(f"   • Caracteres por palabra: {legibilidad.get('caracteres_por_palabra', 'N/A')}")
            print(f"   • Índice de complejidad: {legibilidad.get('indice_complejidad', 'N/A')}")
            print(f"   • Nivel de dificultad: {legibilidad.get('nivel_dificultad', 'N/A')}")
        else:
            print("⚠ Métricas de legibilidad no disponibles")

        # Mostrar información de entidades (si están disponibles)
        entidades = resultados.get('entidades', {})
        if entidades and 'error' not in entidades:
            resumen_ent = entidades.get('resumen', {})
            if resumen_ent:
                print(f"\nENTIDADES ENCONTRADAS:")
                for tipo, cantidad in resumen_ent.items():
                    print(f"   • {tipo}: {cantidad}")
            else:
                print("   • No se encontraron entidades nombradas")
        else:
            print("⚠ Extracción de entidades no disponible (requiere spaCy)")

    except Exception as e:
        print(f"✗ Error mostrando resultados: {e}")

    # Generar visualizaciones
    print(f"\n" + "=" * 50)
    print("GENERANDO VISUALIZACIONES Y REPORTES")
    print("=" * 50)

    try:
        print("Generando visualizaciones...")
        analizador.generar_visualizaciones(guardar=True)
        print("✓ Visualizaciones generadas exitosamente")
    except Exception as e:
        print(f"⚠ Error generando visualizaciones: {e}")
        print("Las visualizaciones pueden requerir dependencias adicionales")

    # Generar reportes
    reportes_generados = []

    try:
        print("Generando reporte HTML...")
        reporte_html = analizador.generar_reporte_completo(formato='html')
        if reporte_html:
            reportes_generados.append(reporte_html)
            print(f"✓ Reporte HTML: {reporte_html}")
    except Exception as e:
        print(f"⚠ Error generando reporte HTML: {e}")

    try:
        print("Generando reporte JSON...")
        reporte_json = analizador.generar_reporte_completo(formato='json')
        if reporte_json:
            reportes_generados.append(reporte_json)
            print(f"✓ Reporte JSON: {reporte_json}")
    except Exception as e:
        print(f"⚠ Error generando reporte JSON: {e}")

    try:
        print("Generando reporte TXT...")
        reporte_txt = analizador.generar_reporte_completo(formato='txt')
        if reporte_txt:
            reportes_generados.append(reporte_txt)
            print(f"✓ Reporte TXT: {reporte_txt}")
    except Exception as e:
        print(f"⚠ Error generando reporte TXT: {e}")

    # Resumen final
    print(f"\n" + "=" * 50)
    print("ANÁLISIS COMPLETADO")
    print("=" * 50)

    if reportes_generados:
        print(f"✓ Se generaron {len(reportes_generados)} reportes:")
        for reporte in reportes_generados:
            print(f"   - {reporte}")
    else:
        print("⚠ No se generaron reportes")

    # Verificar archivos adicionales
    import os
    archivos_adicionales = ['dashboard_analisis_texto.html', 'nube_palabras.png']
    archivos_encontrados = []

    for archivo in archivos_adicionales:
        if os.path.exists(archivo):
            archivos_encontrados.append(archivo)

    if archivos_encontrados:
        print(f"\nArchivos adicionales generados:")
        for archivo in archivos_encontrados:
            print(f"   - {archivo}")

    print(f"\n🎉 ¡Análisis completado exitosamente!")
    print(f"Revisa los archivos generados para ver los resultados detallados.")


def probar_funcionalidades_basicas():
    """
    Prueba las funcionalidades básicas sin generar archivos
    """
    print("\n" + "=" * 60)
    print("PRUEBA RÁPIDA DE FUNCIONALIDADES BÁSICAS")
    print("=" * 60)

    try:
        from analizador_texto_avanzado import AnalizadorTextoAvanzado

        analizador = AnalizadorTextoAvanzado()

        texto_prueba = """
        Este es un texto de prueba para verificar que el analizador funciona correctamente.
        Contiene diferentes tipos de palabras y estructuras para evaluar las capacidades
        del sistema de análisis de texto avanzado.
        """

        print("Ejecutando análisis básico...")
        resultados = analizador.analizar_texto_completo(texto_prueba)

        if resultados:
            resumen = resultados.get('resumen_ejecutivo', {})
            print(f"✓ Palabras analizadas: {resumen.get('total_palabras', 0)}")
            print(f"✓ Diversidad léxica: {resumen.get('diversidad_lexica', 0):.4f}")
            print(f"✓ Sentimiento detectado: {resumen.get('sentimiento_general', 'N/A')}")
            print(f"✓ Análisis completado correctamente")
            return True
        else:
            print("✗ No se generaron resultados")
            return False

    except Exception as e:
        print(f"✗ Error en prueba básica: {e}")
        return False


def verificar_dependencias():
    """
    Verifica que las dependencias estén disponibles
    """
    print("\n" + "=" * 60)
    print("VERIFICACIÓN DE DEPENDENCIAS")
    print("=" * 60)

    dependencias = [
        ('nltk', 'Natural Language Toolkit'),
        ('pandas', 'Data Analysis'),
        ('numpy', 'Numerical Computing'),
        ('matplotlib', 'Plotting'),
        ('seaborn', 'Statistical Visualization'),
        ('wordcloud', 'Word Cloud Generation'),
        ('plotly', 'Interactive Plotting'),
        ('textblob', 'Text Processing'),
    ]

    dependencias_disponibles = 0
    total_dependencias = len(dependencias)

    for modulo, descripcion in dependencias:
        try:
            __import__(modulo)
            print(f"✓ {modulo:<12} - {descripcion}")
            dependencias_disponibles += 1
        except ImportError:
            print(f"✗ {modulo:<12} - NO INSTALADO - {descripcion}")

    # Verificar spaCy por separado
    try:
        import spacy
        print(f"✓ {'spacy':<12} - Advanced NLP")
        dependencias_disponibles += 0.5

        try:
            nlp = spacy.load("es_core_news_sm")
            print(f"✓ {'es_core_news_sm':<12} - Spanish Model")
            dependencias_disponibles += 0.5
        except OSError:
            print(f"⚠ {'es_core_news_sm':<12} - MODELO NO INSTALADO")
            print("  Instala con: python -m spacy download es_core_news_sm")
    except ImportError:
        print(f"✗ {'spacy':<12} - NO INSTALADO - Advanced NLP")

    # Verificar VADER
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print(f"✓ {'vaderSentiment':<12} - Sentiment Analysis")
        dependencias_disponibles += 1
    except ImportError:
        print(f"✗ {'vaderSentiment':<12} - NO INSTALADO - pip install vaderSentiment")

    porcentaje = (dependencias_disponibles / (total_dependencias + 2)) * 100
    print(f"\nDependencias disponibles: {dependencias_disponibles}/{total_dependencias + 2} ({porcentaje:.1f}%)")

    if porcentaje >= 90:
        print("🎉 Todas las dependencias principales están disponibles")
    elif porcentaje >= 70:
        print("⚠ La mayoría de dependencias están disponibles - funcionalidad limitada")
    else:
        print("❌ Muchas dependencias faltan - se requiere instalación")

    return porcentaje >= 70


if __name__ == "__main__":
    try:
        # Verificar dependencias primero
        deps_ok = verificar_dependencias()

        # Prueba básica
        if deps_ok:
            prueba_ok = probar_funcionalidades_basicas()

            if prueba_ok:
                # Ejecutar ejemplo completo
                main()
            else:
                print("\n❌ La prueba básica falló - revisa la configuración")
        else:
            print("\n❌ Faltan dependencias críticas")
            print("Instala las dependencias con:")
            print("pip install nltk pandas numpy matplotlib seaborn wordcloud plotly textblob spacy vaderSentiment")
            print("python -m spacy download es_core_news_sm")

    except KeyboardInterrupt:
        print("\n\nEjecución cancelada por el usuario")
    except Exception as e:
        print(f"\n\nError inesperado: {e}")
        print("Por favor, verifica que todos los archivos estén presentes y las dependencias instaladas")