# Guia Del Proyecto

## 1. Que Hace Este Proyecto

Este proyecto es un framework de investigacion cuantitativa.

Su idea general es:

1. Descargar o cargar datos historicos.
2. Limpiarlos y validarlos.
3. Construir senales cuantitativas.
4. Probar estrategias y parametros.
5. Validar si una senal parece robusta o no.
6. Simular escenarios con Monte Carlo.

No es una interfaz visual todavia. Primero se esta fortaleciendo la logica interna.


## 2. Punto De Entrada

El archivo principal es:

- [run.py](C:/Users/Amaury54/Desktop/stractegy/run.py)

Ese archivo no hace calculos grandes por si mismo. Solo decide que flujo ejecutar.

Comandos principales:

```powershell
python run.py
python run.py demo
python run.py scan
python run.py optuna
python run.py signal momentum_12_1
python run.py portfolio alpha_pro_multi
python run.py full
python run.py mc
```


## 3. Que Hace Cada Modo

### `demo`

Es una corrida corta para comprobar que el framework funciona.

Hace esto:

1. Descarga o reutiliza datos.
2. Construye algunas senales.
3. Investiga unas pocas senales.
4. Corre un mini scanner.
5. Guarda un reporte.

Archivo clave:

- [notebook.py](C:/Users/Amaury54/Desktop/stractegy/research/notebook.py)


### `scan`

Corre un barrido grande de parametros para estrategias.

Sirve para responder:

- que familia de estrategia se ve mejor,
- que combinacion de parametros parece mejor,
- que tan robusta fue en distintos regimenes simulados.

Archivo clave:

- [param_scanner.py](C:/Users/Amaury54/Desktop/stractegy/src/models/param_scanner.py)


### `optuna`

Hace algo parecido a `scan`, pero usando Optuna en vez de probar todo por grilla.

Sirve para buscar parametros de forma mas inteligente.

Archivo clave:

- [optuna_scanner.py](C:/Users/Amaury54/Desktop/stractegy/src/models/optuna_scanner.py)


### `signal NAME`

Analiza una sola senal en profundidad.

Ejemplo:

```powershell
python run.py signal momentum_12_1
```

Sirve para estudiar una senal concreta y validar si parece tener ventaja estadistica.


### `full`

Este es el flujo mas completo del proyecto ahora mismo.

Hace esto:

1. Corre ingestion de datos.
2. Revisa calidad de tickers.
3. Hace un scan de estrategias.
4. Construye varias senales.
5. Evalua esas senales.
6. Guarda un reporte final.

Si quieres ver "casi todo el backend trabajando junto", este es el modo mas cercano.


### `portfolio NAME`

Convierte un alpha en una estrategia de cartera realista.

Hace esto:

1. asegura datos historicos,
2. construye el alpha,
3. selecciona activos por ranking,
4. asigna pesos,
5. rebalancea la cartera,
6. descuenta costos,
7. guarda equity, pesos y reporte.

Ejemplo:

```powershell
python run.py portfolio alpha_pro_multi
```


### `mc`

Corre el modulo de simulacion Monte Carlo.

Sirve para:

- comparar modelos de precios,
- evaluar estrategias bajo simulacion,
- hacer analisis de sensibilidad,
- revisar escenarios sinteticos.

Archivo clave:

- [mc_notebook.py](C:/Users/Amaury54/Desktop/stractegy/research/mc_notebook.py)


## 4. Mapa De Archivos

### Configuracion

- [settings.py](C:/Users/Amaury54/Desktop/stractegy/config/settings.py)

Aqui estan:

- tickers por defecto,
- fechas por defecto,
- configuracion de riesgo,
- configuracion del scan,
- configuracion de Monte Carlo,
- configuracion de proveedor de datos.


### Ingestion De Datos

- [pipeline.py](C:/Users/Amaury54/Desktop/stractegy/src/ingestion/pipeline.py)

Este modulo:

1. descarga datos,
2. los limpia,
3. valida calidad,
4. y guarda parquet.

Capas importantes:

- `OHLCVDownloader`
- `DataCleaner`
- `DataValidator`
- `DataStore`
- `IngestionPipeline`


### Construccion De Senales

- [indicators.py](C:/Users/Amaury54/Desktop/stractegy/src/features/indicators.py)

Aqui viven las senales e indicadores.

Clases importantes:

- `MomentumSignals`
- `MeanReversionSignals`
- `VolatilitySignals`
- `MicrostructureSignals`
- `CrossSectionNormalizer`
- `ICAnalyzer`


### Validacion Estadistica

- [edge_finder.py](C:/Users/Amaury54/Desktop/stractegy/src/models/edge_finder.py)

Este modulo intenta contestar:

- la senal tiene IC positivo,
- aguanta validacion temporal,
- sobrevive BH/FDR,
- parece azar o no,
- se mantiene fuera de muestra.

Clases importantes:

- `EdgeFinder`
- `MultipleHypothesisTester`
- `BootstrapValidator`
- `WalkForwardTester`
- `TimeSeriesValidator`
- `PurgedWalkForwardValidator`


### Scanner De Parametros

- [param_scanner.py](C:/Users/Amaury54/Desktop/stractegy/src/models/param_scanner.py)

Aqui se generan precios sinteticos y se backtestean muchas estrategias con muchos parametros.

Cosas importantes:

- `FAMILIES`
- `run_backtest(...)`
- `ParamScanner`


### Optimizacion Bayesiana

- [optuna_scanner.py](C:/Users/Amaury54/Desktop/stractegy/src/models/optuna_scanner.py)

Es parecido al scanner, pero usa Optuna para buscar mejores parametros.


### Riesgo Y Performance

- [metrics.py](C:/Users/Amaury54/Desktop/stractegy/src/risk/metrics.py)

Aqui estan metricas como:

- Sharpe,
- Sortino,
- CAGR,
- MDD,
- VaR,
- CVaR,
- turnover,
- hit ratio.


### Monte Carlo

- [simulator.py](C:/Users/Amaury54/Desktop/stractegy/src/montecarlo/simulator.py)

Este modulo genera trayectorias simuladas con varios modelos.

Modelos actuales:

- GBM
- Merton
- Heston
- Regime Switching
- Bootstrap
- GARCH


### Ejecucion De Cartera

- [backtester.py](C:/Users/Amaury54/Desktop/stractegy/src/portfolio/backtester.py)

Este modulo convierte un alpha en una estrategia de cartera operable.

Incluye:

- ranking cross-sectional,
- seleccion long/short,
- pesos con ajuste simple por volatilidad,
- rebalanceo,
- costos de transaccion,
- equity curve y reporte de performance.


## 5. Flujo Real Del Backend

Si lo resumimos mucho, el backend trabaja asi:

```text
run.py
  -> research/notebook.py
      -> ingestion/pipeline.py
      -> features/indicators.py
      -> models/edge_finder.py
      -> models/param_scanner.py
      -> models/optuna_scanner.py
      -> montecarlo/simulator.py
```

Un flujo real tipo `full` seria:

```text
run.py full
  -> run_full_pipeline()
  -> IngestionPipeline().run()
  -> ParamScanner(...).scan()
  -> _build_signal(...)
  -> EdgeFinder().batch_investigate(...)
  -> guardar reporte
```


## 6. Donde Se Guardan Cosas

Datos:

- `data/raw/`
- `data/processed/`
- `data/features/`

Reportes:

- `research/reports/`

Logs:

- `logs/`


## 7. Si Quieres Entenderlo En Orden

Te recomiendo leer en este orden:

1. [run.py](C:/Users/Amaury54/Desktop/stractegy/run.py)
2. [notebook.py](C:/Users/Amaury54/Desktop/stractegy/research/notebook.py)
3. [pipeline.py](C:/Users/Amaury54/Desktop/stractegy/src/ingestion/pipeline.py)
4. [indicators.py](C:/Users/Amaury54/Desktop/stractegy/src/features/indicators.py)
5. [edge_finder.py](C:/Users/Amaury54/Desktop/stractegy/src/models/edge_finder.py)
6. [param_scanner.py](C:/Users/Amaury54/Desktop/stractegy/src/models/param_scanner.py)
7. [simulator.py](C:/Users/Amaury54/Desktop/stractegy/src/montecarlo/simulator.py)


## 8. Si Corres El Codigo Y Siempre Ves Lo Mismo

Eso pasa porque:

```python
mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
```

Si ejecutas:

```powershell
python run.py
```

entonces entra en `demo` automaticamente.

Si quieres otro flujo, debes pasar el modo:

```powershell
python run.py full
python run.py scan
python run.py mc
```


## 9. Mi Recomendacion Ahora

Antes de construir interfaz, conviene terminar de estabilizar:

1. ingestion,
2. senales,
3. validacion estadistica,
4. backtests,
5. reportes.

Despues de eso, hacer una interfaz sera mucho mas facil.


## 10. Siguiente Paso Recomendado

Si quieres seguir entendiendo el proyecto sin perderte, el mejor siguiente paso es:

1. documentar tambien [notebook.py](C:/Users/Amaury54/Desktop/stractegy/research/notebook.py),
2. o dibujar un flujo visual simple de `demo`, `full` y `mc`.

Ambos te van a aclarar bastante el panorama.
