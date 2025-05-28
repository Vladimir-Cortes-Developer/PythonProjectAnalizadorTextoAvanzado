# ejemplo_uso.py
from analizador_texto_avanzado import AnalizadorTextoAvanzado


def main():
    # Crear el analizador
    analizador = AnalizadorTextoAvanzado()

    # Tu texto a analizar
    texto = """
    La inteligencia artificial está revolucionando el mundo moderno. 
    Los algoritmos de machine learning permiten a las máquinas aprender 
    de grandes volúmenes de datos sin ser programadas explícitamente. 
    Esta tecnología tiene aplicaciones en medicina, finanzas, educación 
    y muchos otros campos. Sin embargo, también plantea desafíos éticos 
    importantes que debemos considerar cuidadosamente.

    El futuro de la IA depende de cómo desarrollemos estas tecnologías 
    de manera responsable. Es crucial que mantengamos un equilibrio 
    entre la innovación y la protección de los derechos humanos.

    Los sistemas de inteligencia artificial pueden procesar información
    a velocidades que superan las capacidades humanas, pero requieren
    supervisión constante para evitar sesgos y errores. La colaboración
    entre humanos y máquinas será fundamental para el progreso futuro.
    """

    # Realizar análisis completo
    print("Iniciando análisis completo...")
    resultados = analizador.analizar_texto_completo(texto)

    # Mostrar resultados principales
    print("\n" + "=" * 50)
    print("RESULTADOS DEL ANÁLISIS")
    print("=" * 50)

    # Resumen ejecutivo
    resumen = resultados['resumen_ejecutivo']
    print(f"Palabras totales: {resumen['total_palabras']}")
    print(f"Palabras únicas: {resumen['palabras_unicas']}")
    print(f"Diversidad léxica: {resumen['diversidad_lexica']}")
    print(f"Sentimiento: {resumen['sentimiento_general']}")
    print(f"Complejidad: {resumen['complejidad']}")

    print(f"\nTemas principales: {', '.join(resumen['temas_principales'])}")

    # Top 10 palabras más frecuentes
    print(f"\nTOP 10 PALABRAS MÁS FRECUENTES:")
    for i, (palabra, freq) in enumerate(resultados['frecuencias']['mas_frecuentes'][:10], 1):
        print(f"{i:2d}. {palabra:<15} ({freq} veces)")

    # Bigramas importantes
    print(f"\nBIGRAMAS MÁS COMUNES:")
    for i, (bigrama, freq) in enumerate(resultados['bigramas'][:5], 1):
        print(f"{i}. '{' '.join(bigrama)}' ({freq} veces)")

    # Mostrar análisis de sentimientos detallado
    if 'textblob' in resultados['sentimientos']:
        sent = resultados['sentimientos']['textblob']
        print(f"\nANÁLISIS DE SENTIMIENTOS:")
        print(f"Polaridad: {sent['polaridad']} ({sent['clasificacion']})")
        print(f"Subjetividad: {sent['subjetividad']}")

    # Mostrar métricas de legibilidad
    leg = resultados['legibilidad']
    print(f"\nMÉTRICAS DE LEGIBILIDAD:")
    print(f"Oraciones: {leg.get('oraciones', 'N/A')}")
    print(f"Palabras por oración: {leg.get('palabras_por_oracion', 'N/A')}")
    print(f"Caracteres por palabra: {leg.get('caracteres_por_palabra', 'N/A')}")
    print(f"Nivel de dificultad: {leg.get('nivel_dificultad', 'N/A')}")

    # Generar visualizaciones
    print(f"\nGenerando visualizaciones...")
    try:
        analizador.generar_visualizaciones(guardar=True)
        print("Visualizaciones generadas exitosamente")
    except Exception as e:
        print(f"Error generando visualizaciones: {e}")

    # Generar reportes
    print(f"\nGenerando reportes...")
    try:
        reporte_html = analizador.generar_reporte_completo(formato='html')
        reporte_json = analizador.generar_reporte_completo(formato='json')
        reporte_txt = analizador.generar_reporte_completo(formato='txt')

        print(f"\nArchivos generados:")
        print(f"   - {reporte_html}")
        print(f"   - {reporte_json}")
        print(f"   - {reporte_txt}")
        print(f"   - dashboard_analisis_texto.html")
        print(f"   - nube_palabras.png")
    except Exception as e:
        print(f"Error generando reportes: {e}")

    print(f"\nAnálisis completado!")


if __name__ == "__main__":
    main()