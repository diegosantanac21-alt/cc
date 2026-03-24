#!/usr/bin/env python3
"""
Interfaz de escritorio para el framework cuantitativo.

Permite lanzar los flujos principales sin usar la linea de comandos.
"""
import contextlib
import io
import logging
import os
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class QueueWriter(io.TextIOBase):

    def __init__(self, out_queue):
        self.out_queue = out_queue

    def write(self, text):
        if text:
            self.out_queue.put(text)
        return len(text)

    def flush(self):
        return None


class GuiLogHandler(logging.Handler):

    def __init__(self, out_queue):
        super().__init__()
        self.out_queue = out_queue
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.out_queue.put(self.format(record) + "\n")
        except Exception:
            pass


class QuantApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Stractegy - Quant Research Interface")
        self.root.geometry("1180x760")
        self.root.minsize(1000, 680)

        self.output_queue = queue.Queue()
        self.is_running = False
        self.action_buttons = []

        self._build_style()
        self._build_layout()
        self._load_guide()
        self._poll_output_queue()

    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        self.root.configure(bg="#f4efe6")
        style.configure("Main.TFrame", background="#f4efe6")
        style.configure("Card.TFrame", background="#fbf7f0", relief="flat")
        style.configure("Title.TLabel", background="#f4efe6", foreground="#1f2937", font=("Segoe UI Semibold", 18))
        style.configure("Sub.TLabel", background="#f4efe6", foreground="#4b5563", font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background="#fbf7f0", foreground="#111827", font=("Segoe UI Semibold", 12))
        style.configure("CardText.TLabel", background="#fbf7f0", foreground="#374151", font=("Segoe UI", 10))
        style.configure("Run.TButton", font=("Segoe UI Semibold", 10), padding=(10, 8))

    def _build_layout(self):
        outer = ttk.Frame(self.root, style="Main.TFrame", padding=16)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="Main.TFrame")
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="Stractegy Research Console", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Interfaz para correr demo, scan, signal, full y Monte Carlo sobre el backend actual.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, style="Main.TFrame", padding=(0, 0, 12, 0))
        right = ttk.Frame(body, style="Main.TFrame")
        body.add(left, weight=1)
        body.add(right, weight=2)

        self._build_controls(left)
        self._build_output(right)

    def _build_controls(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        quick_tab = ttk.Frame(notebook, style="Card.TFrame", padding=14)
        signal_tab = ttk.Frame(notebook, style="Card.TFrame", padding=14)
        portfolio_tab = ttk.Frame(notebook, style="Card.TFrame", padding=14)
        docs_tab = ttk.Frame(notebook, style="Card.TFrame", padding=14)

        notebook.add(quick_tab, text="Acciones")
        notebook.add(signal_tab, text="Senal")
        notebook.add(portfolio_tab, text="Cartera")
        notebook.add(docs_tab, text="Guia")

        self._build_quick_tab(quick_tab)
        self._build_signal_tab(signal_tab)
        self._build_portfolio_tab(portfolio_tab)
        self._build_docs_tab(docs_tab)

    def _build_quick_tab(self, parent):
        ttk.Label(parent, text="Flujos Principales", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Usa estos botones para ejecutar los modos principales sin escribir comandos.",
            style="CardText.TLabel",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(4, 12))

        self._add_action_button(parent, "Demo", lambda: self._run_task("demo", self._call_run_demo))
        self._add_action_button(parent, "Scan", lambda: self._run_task("scan", self._call_run_param_scan))
        self._add_action_button(parent, "Optuna", lambda: self._run_task("optuna", self._call_run_optuna_scan))
        self._add_action_button(parent, "Full Pipeline", lambda: self._run_task("full", self._call_run_full_pipeline))
        self._add_action_button(parent, "Monte Carlo", lambda: self._run_task("mc", self._call_run_full_mc_report))

        info = (
            "Demo: flujo corto para validar el sistema.\n"
            "Scan: prueba muchas configuraciones por grilla.\n"
            "Optuna: busca parametros automaticamente.\n"
            "Full: ingestion + scan + evaluacion + reporte.\n"
            "Monte Carlo: simulaciones y analisis de escenarios."
        )
        ttk.Label(parent, text=info, style="CardText.TLabel", wraplength=320, justify="left").pack(anchor="w", pady=(14, 0))

    def _build_signal_tab(self, parent):
        ttk.Label(parent, text="Investigar Una Senal", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Escribe el nombre de la senal y corre el analisis estadistico sobre datos historicos.",
            style="CardText.TLabel",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(4, 12))

        form = ttk.Frame(parent, style="Card.TFrame")
        form.pack(fill="x", pady=(0, 10))

        ttk.Label(form, text="Nombre de la senal", style="CardText.TLabel").grid(row=0, column=0, sticky="w")
        self.signal_var = tk.StringVar(value="momentum_12_1")
        signal_entry = ttk.Entry(form, textvariable=self.signal_var, width=28)
        signal_entry.grid(row=1, column=0, sticky="ew", pady=(4, 10))

        ttk.Label(form, text="Sugerencias", style="CardText.TLabel").grid(row=2, column=0, sticky="w")
        suggestions = ttk.Combobox(
            form,
            values=[
                "momentum_12_1",
                "momentum_3m",
                "momentum_1m",
                "trend_ma200",
                "ma_cross_10_50",
                "triple_ma",
                "zscore_20d",
                "zscore_60d",
                "bb_pct_b",
                "rsi_14",
                "rsi_7",
                "low_vol",
                "vol_momentum",
                "trend_rsi_lowvol",
                "alpha_trend_quality",
                "alpha_pullback_trend",
                "alpha_compression_breakout",
                "alpha_pro_multi",
            ],
            textvariable=self.signal_var,
            state="readonly",
        )
        suggestions.grid(row=3, column=0, sticky="ew", pady=(4, 12))

        form.columnconfigure(0, weight=1)

        self._add_action_button(
            parent,
            "Correr Signal",
            lambda: self._run_task(
                f"signal:{self.signal_var.get().strip() or 'momentum_12_1'}",
                self._call_run_signal_investigation,
                self.signal_var.get().strip() or "momentum_12_1",
            ),
        )

        ttk.Label(
            parent,
            text="Este flujo usa research/notebook.py -> run_signal_investigation(...).",
            style="CardText.TLabel",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(12, 0))

    def _build_portfolio_tab(self, parent):
        ttk.Label(parent, text="Estrategia De Cartera", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Convierte un alpha en una estrategia operable con ranking, pesos, rebalanceo y equity curve.",
            style="CardText.TLabel",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(4, 12))

        form = ttk.Frame(parent, style="Card.TFrame")
        form.pack(fill="x", pady=(0, 10))

        ttk.Label(form, text="Alpha de cartera", style="CardText.TLabel").grid(row=0, column=0, sticky="w")
        self.portfolio_alpha_var = tk.StringVar(value="alpha_pro_multi")
        combo = ttk.Combobox(
            form,
            values=[
                "alpha_trend_quality",
                "alpha_pullback_trend",
                "alpha_compression_breakout",
                "alpha_pro_multi",
                "trend_rsi_lowvol",
            ],
            textvariable=self.portfolio_alpha_var,
            state="readonly",
        )
        combo.grid(row=1, column=0, sticky="ew", pady=(4, 12))
        form.columnconfigure(0, weight=1)

        self._add_action_button(
            parent,
            "Correr Portfolio",
            lambda: self._run_task(
                f"portfolio:{self.portfolio_alpha_var.get().strip() or 'alpha_pro_multi'}",
                self._call_run_portfolio_alpha_strategy,
                self.portfolio_alpha_var.get().strip() or "alpha_pro_multi",
            ),
        )

        text = (
            "Este flujo usa ranking cross-sectional, pesos por lado, rebalanceo\n"
            "y costos de transaccion para convertir el alpha en estrategia."
        )
        ttk.Label(parent, text=text, style="CardText.TLabel", wraplength=320, justify="left").pack(anchor="w", pady=(12, 0))

    def _build_docs_tab(self, parent):
        ttk.Label(parent, text="Guia Del Proyecto", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Resumen rapido del proyecto, mapa de archivos y explicacion de modos.",
            style="CardText.TLabel",
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        self.docs_box = ScrolledText(
            parent,
            wrap="word",
            height=20,
            font=("Consolas", 10),
            bg="#fffdf8",
            fg="#1f2937",
            insertbackground="#1f2937",
        )
        self.docs_box.pack(fill="both", expand=True)
        self.docs_box.configure(state="disabled")

    def _build_output(self, parent):
        top = ttk.Frame(parent, style="Main.TFrame")
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Salida", style="CardTitle.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(top, textvariable=self.status_var, style="Sub.TLabel").pack(side="right")

        self.output_box = ScrolledText(
            parent,
            wrap="word",
            font=("Consolas", 10),
            bg="#15202b",
            fg="#f8fafc",
            insertbackground="#f8fafc",
        )
        self.output_box.pack(fill="both", expand=True)

        bottom = ttk.Frame(parent, style="Main.TFrame")
        bottom.pack(fill="x", pady=(8, 0))

        clear_btn = ttk.Button(bottom, text="Limpiar Salida", command=self._clear_output, style="Run.TButton")
        clear_btn.pack(side="left")
        self.action_buttons.append(clear_btn)

    def _add_action_button(self, parent, text, command):
        btn = ttk.Button(parent, text=text, command=command, style="Run.TButton")
        btn.pack(fill="x", pady=4)
        self.action_buttons.append(btn)
        return btn

    def _append_output(self, text):
        self.output_box.insert("end", text)
        self.output_box.see("end")

    def _clear_output(self):
        self.output_box.delete("1.0", "end")

    def _load_guide(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GUIA_PROYECTO.md")
        content = "No se encontro GUIA_PROYECTO.md."
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        self.docs_box.configure(state="normal")
        self.docs_box.delete("1.0", "end")
        self.docs_box.insert("1.0", content)
        self.docs_box.configure(state="disabled")

    def _set_running_state(self, running, label="Listo"):
        self.is_running = running
        self.status_var.set(label)
        state = "disabled" if running else "normal"
        for btn in self.action_buttons:
            btn.configure(state=state)

    def _call_run_demo(self):
        from research.notebook import run_demo
        return run_demo()

    def _call_run_param_scan(self):
        from research.notebook import run_param_scan
        return run_param_scan()

    def _call_run_optuna_scan(self):
        from research.notebook import run_optuna_scan
        return run_optuna_scan()

    def _call_run_full_pipeline(self):
        from research.notebook import run_full_pipeline
        return run_full_pipeline()

    def _call_run_signal_investigation(self, signal_name):
        from research.notebook import run_signal_investigation
        return run_signal_investigation(signal_name)

    def _call_run_portfolio_alpha_strategy(self, alpha_name):
        from research.notebook import run_portfolio_alpha_strategy
        return run_portfolio_alpha_strategy(alpha_name)

    def _call_run_full_mc_report(self):
        from research.mc_notebook import run_full_mc_report
        return run_full_mc_report()

    def _run_task(self, label, func, *args, **kwargs):
        if self.is_running:
            self._append_output("\n[INFO] Ya hay una tarea corriendo. Espera a que termine.\n")
            return

        self._append_output(f"\n{'=' * 72}\n")
        self._append_output(f"Iniciando: {label}\n")
        self._append_output(f"{'=' * 72}\n")
        self._set_running_state(True, f"Corriendo {label}...")

        thread = threading.Thread(
            target=self._task_wrapper,
            args=(label, func, args, kwargs),
            daemon=True,
        )
        thread.start()

    def _task_wrapper(self, label, func, args, kwargs):
        writer = QueueWriter(self.output_queue)
        handler = GuiLogHandler(self.output_queue)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                result = func(*args, **kwargs)
                if result is not None:
                    print(f"\n[OK] {label} finalizado.")
        except Exception:
            self.output_queue.put("\n[ERROR] La tarea fallo.\n")
            self.output_queue.put(traceback.format_exc() + "\n")
        finally:
            root_logger.removeHandler(handler)
            self.output_queue.put(("__STATUS__", f"Listo"))
            self.output_queue.put(("__DONE__", None))

    def _poll_output_queue(self):
        try:
            while True:
                item = self.output_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__DONE__":
                    self._set_running_state(False, "Listo")
                elif isinstance(item, tuple) and item[0] == "__STATUS__":
                    self.status_var.set(item[1])
                else:
                    self._append_output(str(item))
        except queue.Empty:
            pass
        self.root.after(120, self._poll_output_queue)


def launch_gui():
    root = tk.Tk()
    app = QuantApp(root)
    root.mainloop()
    return app


if __name__ == "__main__":
    launch_gui()
