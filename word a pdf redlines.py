"""
Convierte todos los archivos de Word (.docx, .doc, .docm) de una carpeta a PDF
CONSERVANDO los redlines (control de cambios) y los comentarios.

Requisitos:
    - Windows con Microsoft Word instalado
    - pip install pywin32

Uso:
    python word_a_pdf_redlines.py "C:\\ruta\\a\\carpeta"
    python word_a_pdf_redlines.py "C:\\ruta\\a\\carpeta" --salida "C:\\ruta\\pdfs" --subcarpetas
"""

import argparse
import sys
from pathlib import Path

import pythoncom
import win32com.client as win32

# Constantes de Word
WD_EXPORT_FORMAT_PDF = 17
WD_EXPORT_DOCUMENT_WITH_MARKUP = 7      # Exporta CON el marcado (redlines)
WD_EXPORT_OPTIMIZE_FOR_PRINT = 0
WD_REVISIONS_VIEW_FINAL = 0             # Vista "Final" + marcado
WD_REVISIONS_MARKUP_ALL = 2             # "Todas las marcas"
WD_BALLOON_REVISIONS = 0                # Cambios en globos al margen
WD_INLINE_REVISIONS = 1                 # Cambios en línea (tachado/subrayado)
WD_DO_NOT_SAVE_CHANGES = 0

EXTENSIONES = {".docx", ".doc", ".docm"}


def preparar_vista_con_marcas(doc, en_linea: bool):
    """Configura la vista del documento para que se vean todos los redlines."""
    vista = doc.Windows(1).View
    vista.ShowRevisionsAndComments = True
    vista.RevisionsView = WD_REVISIONS_VIEW_FINAL
    try:
        vista.RevisionsFilter.Markup = WD_REVISIONS_MARKUP_ALL   # Word 2013+
    except Exception:
        pass
    try:
        vista.MarkupMode = WD_INLINE_REVISIONS if en_linea else WD_BALLOON_REVISIONS
    except Exception:
        pass


def convertir_carpeta(carpeta: Path, salida: Path, subcarpetas: bool, en_linea: bool):
    patron = "**/*" if subcarpetas else "*"
    archivos = [
        f for f in carpeta.glob(patron)
        if f.is_file()
        and f.suffix.lower() in EXTENSIONES
        and not f.name.startswith("~$")      # ignora archivos temporales de Word
    ]

    if not archivos:
        print("No se encontraron archivos de Word en la carpeta.")
        return

    pythoncom.CoInitialize()
    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0

    ok, errores = 0, []
    try:
        for archivo in archivos:
            relativo = archivo.relative_to(carpeta)
            destino = (salida / relativo).with_suffix(".pdf")
            destino.parent.mkdir(parents=True, exist_ok=True)

            print(f"Convirtiendo: {relativo} ...", end=" ")
            doc = None
            try:
                doc = word.Documents.Open(
                    str(archivo.resolve()),
                    ConfirmConversions=False,
                    ReadOnly=True,
                    AddToRecentFiles=False,
                )
                preparar_vista_con_marcas(doc, en_linea)

                doc.ExportAsFixedFormat(
                    OutputFileName=str(destino.resolve()),
                    ExportFormat=WD_EXPORT_FORMAT_PDF,
                    OpenAfterExport=False,
                    OptimizeFor=WD_EXPORT_OPTIMIZE_FOR_PRINT,
                    Item=WD_EXPORT_DOCUMENT_WITH_MARKUP,
                    IncludeDocProps=True,
                    CreateBookmarks=1,          # marcadores desde los títulos
                    DocStructureTags=True,
                )
                ok += 1
                print("OK")
            except Exception as e:
                errores.append((relativo, e))
                print(f"ERROR: {e}")
            finally:
                if doc is not None:
                    doc.Close(SaveChanges=WD_DO_NOT_SAVE_CHANGES)
    finally:
        word.Quit()
        pythoncom.CoUninitialize()

    print(f"\nListo: {ok} convertidos, {len(errores)} con error.")
    for nombre, e in errores:
        print(f"  - {nombre}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Word a PDF conservando redlines")
    parser.add_argument("carpeta", help="Carpeta con los archivos de Word")
    parser.add_argument("--salida", help="Carpeta destino (por defecto: <carpeta>\\PDF)")
    parser.add_argument("--subcarpetas", action="store_true", help="Incluir subcarpetas")
    parser.add_argument("--en-linea", action="store_true",
                        help="Mostrar cambios en línea en vez de globos al margen")
    args = parser.parse_args()

    carpeta = Path(args.carpeta)
    if not carpeta.is_dir():
        sys.exit(f"La carpeta no existe: {carpeta}")

    salida = Path(args.salida) if args.salida else carpeta / "PDF"
    convertir_carpeta(carpeta, salida, args.subcarpetas, args.en_linea)


if __name__ == "__main__":
    main()
