# fix_nltk.py - Script para arreglar problemas de NLTK
import nltk
import ssl


def descargar_recursos_nltk():
    """Descarga todos los recursos necesarios de NLTK"""

    # Configurar SSL para evitar errores de certificado
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    print("Descargando recursos de NLTK...")

    # Lista completa de recursos necesarios (incluyendo los nuevos)
    recursos = [
        'punkt',
        'punkt_tab',  # Nuevo recurso requerido
        'stopwords',
        'vader_lexicon',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4',
        'brown',
        'names',
        'universal_tagset'
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
        print(f"\nRecursos que fallaron: {errores}")
        print("Intentando descarga alternativa...")

        # Descarga alternativa para recursos problemáticos
        for recurso in errores:
            try:
                print(f"Intento alternativo para {recurso}...")
                nltk.download(recurso, download_dir=None, quiet=False)
                print(f"✓ {recurso} descargado (intento alternativo)")
            except Exception as e:
                print(f"✗ Falló intento alternativo para {recurso}: {e}")

    print("\nProceso de descarga completado.")

    # Verificar que los recursos están disponibles
    verificar_recursos()


def verificar_recursos():
    """Verifica que los recursos estén correctamente instalados"""
    print("\nVerificando recursos...")

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

        print("\n✅ Todos los recursos funcionan correctamente!")
        return True

    except Exception as e:
        print(f"\n❌ Error verificando recursos: {e}")
        return False


def descargar_todo():
    """Descarga TODO lo disponible en NLTK (solución nuclear)"""
    print("Descargando TODOS los recursos de NLTK...")
    print("Esto puede tomar varios minutos...")

    try:
        nltk.download('all')
        print("✓ Descarga completa exitosa")
    except Exception as e:
        print(f"✗ Error en descarga completa: {e}")


def main():
    print("SOLUCIONADOR DE PROBLEMAS NLTK")
    print("=" * 40)
    print("1. Descargar recursos específicos (recomendado)")
    print("2. Descargar TODO (toma más tiempo)")
    print("3. Solo verificar recursos existentes")

    opcion = input("\nSelecciona una opción (1-3): ").strip()

    if opcion == "1":
        descargar_recursos_nltk()
    elif opcion == "2":
        descargar_todo()
    elif opcion == "3":
        verificar_recursos()
    else:
        print("Opción no válida, ejecutando descarga específica...")
        descargar_recursos_nltk()


if __name__ == "__main__":
    main()