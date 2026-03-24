#!/usr/bin/env python3
"""
Punto de entrada del proyecto.

Modos disponibles:
- demo: ejecuta una demostracion compacta del framework.
- scan: corre el barrido de parametros por grilla.
- optuna: busca parametros con Optuna.
- signal NAME: investiga una senal puntual.
- portfolio NAME: convierte un alpha en una estrategia de cartera.
- full: ejecuta el pipeline completo.
- mc: genera el reporte de simulacion Monte Carlo.
- ui: abre una interfaz de escritorio.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


MODE_DESCRIPTIONS = {
    "demo": "Demo general del framework con descarga, senales y mini scan.",
    "scan": "Grid search de parametros sobre familias de estrategias.",
    "optuna": "Optimizacion bayesiana de parametros con Optuna.",
    "signal": "Investigacion detallada de una senal especifica.",
    "portfolio": "Backtest de cartera sobre un alpha con ranking, pesos y rebalanceo.",
    "full": "Pipeline completo: descarga datos, corre scan, evalua senales y guarda reporte.",
    "mc": "Reporte completo de simulacion Monte Carlo.",
    "ui": "Interfaz grafica de escritorio para usar el backend sin comandos.",
}


def print_help():
    print("Uso: python run.py [modo]")
    print("")
    print("Modos disponibles:")
    print("  demo           ->", MODE_DESCRIPTIONS["demo"])
    print("  scan           ->", MODE_DESCRIPTIONS["scan"])
    print("  optuna         ->", MODE_DESCRIPTIONS["optuna"])
    print("  signal NAME    ->", MODE_DESCRIPTIONS["signal"])
    print("  portfolio NAME ->", MODE_DESCRIPTIONS["portfolio"])
    print("  full           ->", MODE_DESCRIPTIONS["full"])
    print("  mc             ->", MODE_DESCRIPTIONS["mc"])
    print("  ui             ->", MODE_DESCRIPTIONS["ui"])
    print("")
    print("Ejemplos:")
    print("  python run.py")
    print("  python run.py scan")
    print("  python run.py signal momentum_12_1")
    print("  python run.py portfolio alpha_pro_multi")
    print("  python run.py full")
    print("  python run.py mc")
    print("  python run.py ui")
    print("")
    print("Ejemplo del modo 'full':")
    print("  1. Revisa/descarga datos historicos.")
    print("  2. Filtra tickers con calidad suficiente.")
    print("  3. Corre el scanner de estrategias.")
    print("  4. Evalua varias senales cuantitativas.")
    print("  5. Guarda un reporte en research/reports/full_pipeline_report.txt")

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode == "demo":
        # Flujo corto para validar que el framework corre de punta a punta.
        from research.notebook import run_demo; run_demo()
    elif mode == "scan":
        # Barrido exhaustivo de combinaciones de parametros.
        from research.notebook import run_param_scan; run_param_scan()
    elif mode == "optuna":
        # Optimizacion probabilistica para buscar mejores parametros.
        from research.notebook import run_optuna_scan; run_optuna_scan()
    elif mode == "signal":
        # Analiza una senal individual y evalua su robustez estadistica.
        name = sys.argv[2] if len(sys.argv) > 2 else "momentum_12_1"
        from research.notebook import run_signal_investigation
        run_signal_investigation(name)
    elif mode == "portfolio":
        # Convierte un alpha en una cartera rebalanceada con pesos y equity curve.
        name = sys.argv[2] if len(sys.argv) > 2 else "alpha_pro_multi"
        from research.notebook import run_portfolio_alpha_strategy
        run_portfolio_alpha_strategy(name)
    elif mode == "full":
        # Ejemplo completo del proyecto:
        # 1) ingestion de datos
        # 2) scan de estrategias
        # 3) investigacion de senales
        # 4) guardado de reporte final
        from research.notebook import run_full_pipeline; run_full_pipeline()
    elif mode == "mc":
        # Genera escenarios simulados con varios modelos Monte Carlo.
        from research.mc_notebook import run_full_mc_report; run_full_mc_report()
    elif mode == "ui":
        # Abre una interfaz de escritorio para operar el proyecto visualmente.
        from gui_app import launch_gui; launch_gui()
    elif mode in {"help", "--help", "-h"}:
        print_help()
    else:
        print(f"Modo no reconocido: {mode}\n")
        print_help()

if __name__ == "__main__":
    main()
