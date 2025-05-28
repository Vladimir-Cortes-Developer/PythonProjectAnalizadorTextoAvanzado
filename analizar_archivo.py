# analizar_archivo.py
from analizador_texto_avanzado import AnalizadorTextoAvanzado, procesar_texto_desde_archivo, \
    procesar_multiples_archivos, comparar_textos, generar_reporte_comparativo
import os


def analizar_archivo_individual(ruta_archivo):
    """
    Analiza un archivo de texto individual
    """
    # Verificar que el archivo existe
    if not os.path.exists(ruta_archivo):
        print(f"Error: El archivo {ruta_archivo} no existe")
        return False

    # Leer el archivo
    texto = procesar_texto_desde_archivo(ruta_archivo)
    if not texto:
        return False

    print(f"Archivo leído: {ruta_archivo}")
    print(f"Tamaño: {len(texto)} caracteres")

    # Crear analizador
    analizador = AnalizadorTextoAvanzado()

    # Analizar
    resultados = analizador.analizar_texto_completo(texto)

    # Mostrar resultados clave
    print("\n" + "=" * 60)
    print("ANÁLISIS DE ARCHIVO COMPLETADO")
    print("=" * 60)

    # Información general
    resumen = resultados['resumen_ejecutivo']
    print(f"\nESTADÍSTICAS GENERALES:")
    print(f"   • Total de palabras: {resumen['total_palabras']:,}")
    print(f"   • Palabras únicas: {resumen['palabras_unicas']:,}")
    print(f"   • Diversidad léxica: {resumen['diversidad_lexica']:.4f}")
    print(f"   • Sentimiento predominante: {resumen['sentimiento_general']}")
    print(f"   • Nivel de complejidad: {resumen['complejidad']}")

    # Legibilidad detallada
    leg = resultados['legibilidad']
    print(f"\nMÉTRICAS DE LEGIBILIDAD:")
    print(f"   • Oraciones: {leg.get('oraciones', 'N/A')}")
    print(f"   • Palabras por oración: {leg.get('palabras_por_oracion', 'N/A')}")
    print(f"   • Caracteres por palabra: {leg.get('caracteres_por_palabra', 'N/A')}")

    # Temas principales
    print(f"\nTEMAS PRINCIPALES:")
    for i, tema in enumerate(resumen['temas_principales'][:8], 1):
        print(f"   {i}. {tema}")

    # Top 10 palabras más frecuentes
    print(f"\nTOP 10 PALABRAS MÁS FRECUENTES:")
    for i, (palabra, freq) in enumerate(resultados['frecuencias']['mas_frecuentes'][:10], 1):
        print(f"   {i:2d}. {palabra:<15} {freq:>3d} veces")

    # Bigramas
    if resultados['bigramas']:
        print(f"\nTOP 5 BIGRAMAS:")
        for i, (bigrama, freq) in enumerate(resultados['bigramas'][:5], 1):
            print(f"   {i}. '{' '.join(bigrama)}' ({freq} veces)")

    # Entidades (si están disponibles)
    if 'entidades' in resultados and 'resumen' in resultados['entidades']:
        print(f"\nENTIDADES ENCONTRADAS:")
        for tipo, cantidad in resultados['entidades']['resumen'].items():
            print(f"   • {tipo}: {cantidad}")

    # Generar reportes y visualizaciones
    try:
        print(f"\nGenerando visualizaciones y reportes...")
        analizador.generar_visualizaciones(guardar=True)

        reporte_html = analizador.generar_reporte_completo(formato='html')
        reporte_json = analizador.generar_reporte_completo(formato='json')
        reporte_txt = analizador.generar_reporte_completo(formato='txt')

        print(f"\nANÁLISIS COMPLETADO EXITOSAMENTE")
        print(f"Archivos generados:")
        print(f"   {reporte_html}")
        print(f"   {reporte_json}")
        print(f"   {reporte_txt}")
        print(f"   dashboard_analisis_texto.html")
        print(f"   nube_palabras.png")

        return True
    except Exception as e:
        print(f"Error generando archivos: {e}")
        return False


def analizar_carpeta_archivos(carpeta):
    """
    Analiza múltiples archivos en una carpeta
    """
    if not os.path.exists(carpeta):
        print(f"Error: La carpeta {carpeta} no existe")
        return

    extensiones_texto = ['.txt', '.md', '.rst', '.text']
    archivos_encontrados = []

    for archivo in os.listdir(carpeta):
        if any(archivo.lower().endswith(ext) for ext in extensiones_texto):
            archivos_encontrados.append(os.path.join(carpeta, archivo))

    if not archivos_encontrados:
        print(f"No se encontraron archivos de texto en {carpeta}")
        return

    print(f"Encontrados {len(archivos_encontrados)} archivos:")
    for archivo in archivos_encontrados:
        print(f"   • {os.path.basename(archivo)}")

    # Preguntar si analizar individualmente o comparar
    print(f"\nOpciones:")
    print(f"1. Analizar cada archivo individualmente")
    print(f"2. Realizar análisis comparativo")

    opcion = input("Selecciona una opción (1-2): ").strip()

    if opcion == "1":
        # Analizar cada archivo individualmente
        for i, archivo in enumerate(archivos_encontrados, 1):
            print(f"\n{'=' * 60}")
            print(f"ANALIZANDO ARCHIVO {i}/{len(archivos_encontrados)}")
            print(f"{'=' * 60}")
            analizar_archivo_individual(archivo)

    elif opcion == "2":
        # Análisis comparativo
        print(f"\nRealizando análisis comparativo...")

        textos_contenido = {}
        for archivo in archivos_encontrados:
            nombre_archivo = os.path.basename(archivo)
            contenido = procesar_texto_desde_archivo(archivo)
            if contenido:
                textos_contenido[nombre_archivo] = contenido

        if textos_contenido:
            try:
                resultados_comparacion = comparar_textos(textos_contenido)

                # Mostrar resumen comparativo
                print(f"\n{'=' * 60}")
                print(f"ANÁLISIS COMPARATIVO COMPLETADO")
                print(f"{'=' * 60}")

                metricas = resultados_comparacion['analisis_comparativo']['metricas_por_texto']

                print(f"\nRESUMEN POR ARCHIVO:")
                for nombre, datos in metricas.items():
                    print(f"\n{nombre}:")
                    print(f"   Palabras: {datos['palabras_totales']:,}")
                    print(f"   Diversidad: {datos['diversidad_lexica']:.4f}")
                    print(f"   Sentimiento: {datos['sentimiento']}")
                    print(f"   Complejidad: {datos['complejidad']}")

                # Rankings
                print(f"\nRANKING POR COMPLEJIDAD:")
                for i, (nombre, datos) in enumerate(
                        resultados_comparacion['analisis_comparativo']['ranking_complejidad'], 1):
                    print(f"   {i}. {nombre} - {datos['complejidad']}")

                print(f"\nRANKING POR DIVERSIDAD LÉXICA:")
                for i, (nombre, datos) in enumerate(
                        resultados_comparacion['analisis_comparativo']['ranking_diversidad'], 1):
                    print(f"   {i}. {nombre} - {datos['diversidad_lexica']:.4f}")

                # Palabras comunes
                palabras_comunes = resultados_comparacion['analisis_comparativo']['palabras_comunes']
                if palabras_comunes:
                    print(f"\nPALABRAS COMUNES:")
                    print(f"   {', '.join(palabras_comunes)}")

                # Generar reporte comparativo
                reporte_comp = generar_reporte_comparativo(resultados_comparacion, 'html')
                print(f"\nReporte comparativo generado: {reporte_comp}")

            except Exception as e:
                print(f"Error en análisis comparativo: {e}")

    else:
        print("Opción no válida")


def analizar_texto_directo():
    """
    Permite al usuario ingresar texto directamente
    """
    print("Ingresa tu texto (presiona Enter dos veces para terminar):")
    print("=" * 50)

    lineas = []
    lineas_vacias_consecutivas = 0

    while True:
        try:
            linea = input()
            if linea.strip() == "":
                lineas_vacias_consecutivas += 1
                if lineas_vacias_consecutivas >= 2:
                    break
            else:
                lineas_vacias_consecutivas = 0
            lineas.append(linea)
        except (EOFError, KeyboardInterrupt):
            break

    texto = "\n".join(lineas).strip()

    if not texto:
        print("No se ingresó texto para analizar")
        return

    print(f"\nTexto ingresado ({len(texto)} caracteres)")
    print("=" * 50)

    # Crear analizador y procesar
    analizador = AnalizadorTextoAvanzado()

    try:
        resultados = analizador.analizar_texto_completo(texto)

        # Mostrar resumen
        resumen = resultados['resumen_ejecutivo']
        print(f"\nRESULTADOS:")
        print(f"   Palabras: {resumen['total_palabras']}")
        print(f"   Diversidad: {resumen['diversidad_lexica']:.4f}")
        print(f"   Sentimiento: {resumen['sentimiento_general']}")
        print(f"   Complejidad: {resumen['complejidad']}")

        # Generar archivos
        analizador.generar_visualizaciones(guardar=True)
        reporte = analizador.generar_reporte_completo(formato='html')
        print(f"\nReporte generado: {reporte}")

    except Exception as e:
        print(f"Error durante el análisis: {e}")


def mostrar_menu():
    """
    Muestra el menú principal
    """
    print("\n" + "=" * 60)
    print("ANALIZADOR DE TEXTO AVANZADO")
    print("=" * 60)
    print("1. Analizar un archivo específico")
    print("2. Analizar archivos en una carpeta")
    print("3. Analizar texto ingresado directamente")
    print("4. Ayuda y ejemplos")
    print("5. Salir")
    print("=" * 60)


def mostrar_ayuda():
    """
    Muestra información de ayuda
    """
    print("\n" + "=" * 60)
    print("AYUDA - ANALIZADOR DE TEXTO AVANZADO")
    print("=" * 60)

    print("\nFORMATOS DE ARCHIVO SOPORTADOS:")
    print("   • .txt - Archivos de texto plano")
    print("   • .md  - Archivos Markdown")
    print("   • .rst - Archivos reStructuredText")
    print("   • .text - Archivos de texto")

    print("\nTIPOS DE ANÁLISIS:")
    print("   • Frecuencia de palabras y n-gramas")
    print("   • Análisis de sentimientos")
    print("   • Extracción de entidades nombradas")
    print("   • Métricas de legibilidad")
    print("   • Análisis comparativo entre textos")

    print("\nARCHIVOS GENERADOS:")
    print("   • Reportes HTML (visualizable en navegador)")
    print("   • Reportes JSON (datos estructurados)")
    print("   • Reportes TXT (texto plano)")
    print("   • Dashboard interactivo (HTML)")
    print("   • Nube de palabras (PNG)")

    print("\nEJEMPLOS DE USO:")
    print("   • Analizar una novela o ensayo")
    print("   • Comparar diferentes versiones de documentos")
    print("   • Evaluar la complejidad de textos educativos")
    print("   • Análisis de sentimientos en reseñas")

    print("\nREQUISITOS:")
    print("   • Python 3.7+")
    print("   • Librerías: nltk, pandas, matplotlib, etc.")
    print("   • Modelo spaCy (opcional)")


def main():
    """
    Función principal
    """
    while True:
        mostrar_menu()

        try:
            opcion = input("\nSelecciona una opción (1-5): ").strip()

            if opcion == "1":
                ruta = input("Ingresa la ruta del archivo: ").strip().strip('"\'')
                if ruta:
                    analizar_archivo_individual(ruta)
                else:
                    print("Ruta no válida")

            elif opcion == "2":
                carpeta = input("Ingresa la ruta de la carpeta: ").strip().strip('"\'')
                if carpeta:
                    analizar_carpeta_archivos(carpeta)
                else:
                    print("Ruta no válida")

            elif opcion == "3":
                analizar_texto_directo()

            elif opcion == "4":
                mostrar_ayuda()

            elif opcion == "5":
                print("Saliendo del analizador...")
                break

            else:
                print("Opción no válida. Selecciona 1-5.")

        except KeyboardInterrupt:
            print("\n\nSaliendo del analizador...")
            break
        except Exception as e:
            print(f"Error inesperado: {e}")

        # Pausa antes de mostrar el menú nuevamente
        input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    main()