# Crypto Price Forecasting Using Autoformer

Proyecto experimental de prediccion del precio de cierre de Bitcoin (BTC-USD) mediante tecnicas de series temporales, indicadores tecnicos y modelos basados en deep learning. El repositorio documenta el flujo completo desde la descarga de datos historicos hasta la visualizacion de predicciones y la comparacion de errores.

> Este proyecto tiene fines academicos y de investigacion. No constituye asesoria financiera ni debe utilizarse como unica base para tomar decisiones de inversion.

## Objetivo

El objetivo principal es construir y evaluar un pipeline de forecasting para BTC-USD que permita:

- Descargar y preparar datos historicos de Bitcoin desde Yahoo Finance.
- Analizar la evolucion del precio de cierre.
- Calcular indicadores tecnicos como SMA, EMA, RSI y MACD.
- Preparar ventanas temporales para entrenamiento supervisado.
- Entrenar modelos de prediccion sobre series temporales.
- Comparar predicciones contra precios reales en un conjunto de prueba.
- Evaluar el rendimiento con metricas como MSE, MAE, RMSE y R2.

## Vista general

El notebook principal es:

[`Crypto price forecasting using Autoformer.ipynb`](Crypto%20price%20forecasting%20using%20Autoformer.ipynb)

El flujo desarrollado incluye:

1. Descarga de datos de `BTC-USD` con `yfinance`.
2. Limpieza y seleccion de variables relevantes.
3. Visualizacion del precio de cierre historico.
4. Calculo de indicadores tecnicos.
5. Normalizacion de datos con `MinMaxScaler`.
6. Construccion de ventanas temporales con lookback de 7 dias.
7. Division temporal train/test sin barajar los datos.
8. Entrenamiento de un modelo recurrente en PyTorch.
9. Forecasting adicional con `amazon/chronos-t5-tiny`.
10. Comparacion visual y numerica de resultados.

## Dataset

El archivo incluido [`BTC-USD.csv`](BTC-USD.csv) contiene datos diarios de Bitcoin obtenidos desde Yahoo Finance.

- Activo: `BTC-USD`
- Frecuencia: diaria
- Rango incluido en el CSV: 2014-09-17 a 2024-07-08
- Variables originales: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- Variable objetivo: `Close`

![Vista inicial del dataset](BTC-USD%20Head.png)

## Analisis exploratorio

El proyecto incluye graficas para revisar la tendencia historica de BTC y el comportamiento de indicadores tecnicos.

![Precio de cierre BTC](BTC%20Close%20Price.png)

Indicadores calculados:

- Media movil simple: `SMA`
- Medias moviles exponenciales: `EMA 20`, `EMA 50`, `EMA 200`
- Relative Strength Index: `RSI`
- Moving Average Convergence Divergence: `MACD` y `MACD Signal`

![Indicadores tecnicos](Indicators.png)

## Modelos y experimentos

### Modelo recurrente en PyTorch

Se construye una arquitectura recurrente con `torch.nn.LSTM` para modelar dependencias temporales a partir de ventanas de datos normalizadas. El entrenamiento usa:

- Funcion de perdida: `MSELoss`
- Optimizador: `Adam`
- Learning rate: `0.001`
- Epocas: `30`
- Batch size: `16`
- Lookback: `7` dias

### Forecasting con Chronos

Tambien se incluye una prueba con `amazon/chronos-t5-tiny`, usando `ChronosPipeline` para generar un horizonte de prediccion de 31 dias y un intervalo predictivo basado en cuantiles.

![Prediccion Chronos](actual%20price%20vs%20predicted%20price%20test%20%28Chronos%29.png)

### Referencia Autoformer

El notebook conserva un comando de entrenamiento de referencia para Autoformer con una configuracion de horizonte de 31 dias:

```bash
python -u BTC-FORECAST.py \
  --is_training 1 \
  --root_path ./dataset/Cryto/ \
  --data_path BTC-USD.csv \
  --model_id BTC_96_31 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 31 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des BTC_Price_Prediction \
  --itr 1 \
  --train_epochs 100
```

## Resultados obtenidos

En el conjunto de prueba de 31 dias, el experimento principal obtuvo:

| Metrica | Valor |
| --- | ---: |
| MSE | 2,326,499.88 |
| RMSE | 1,525.29 |
| MAE | 1,273.61 |
| R2 | 0.8514 |

Comparacion registrada en el notebook:

| Modelo | MSE | MAE |
| --- | ---: | ---: |
| Autoformer | 2,326,499.88 | 1,273.61 |
| LSTM | 96,413,480.00 | 8,289.03 |

![Tabla de errores](MAE%20MSE%20Table%20.png)

![Grafico de errores](MAE%20MSE%20Chart.png)

Visualizacion de precios reales frente a predicciones:

![Prediccion en entrenamiento](actual%20price%20vs%20predicted%20price.png)

![Prediccion en test](actual%20price%20vs%20predicted%20price%20test.png)

![Comparacion Autoformer LSTM](AUTOFORMER%20LSTM.png)

## Estructura del repositorio

```text
.
├── BTC-USD.csv
├── Crypto price forecasting using Autoformer.ipynb
├── *.png
├── PDF/
│   └── Articulos y documentos de referencia
└── Price-Forecasting-Chronos-T5
```

## Instalacion

Se recomienda usar un entorno virtual de Python.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Instala las dependencias principales:

```bash
pip install notebook pandas numpy matplotlib seaborn scikit-learn torch transformers yfinance ta chronos-forecasting
```

Si se ejecuta Chronos en GPU, verifica que la instalacion de PyTorch sea compatible con tu version de CUDA.

## Uso

1. Clona el repositorio:

```bash
git clone https://github.com/tu-usuario/Crypto-Price-Forecasting-Using-Autoformer.git
cd Crypto-Price-Forecasting-Using-Autoformer
```

2. Abre el notebook:

```bash
jupyter notebook "Crypto price forecasting using Autoformer.ipynb"
```

3. Ejecuta las celdas en orden para:

- Descargar o cargar `BTC-USD.csv`.
- Calcular indicadores tecnicos.
- Preparar los datos de entrenamiento y prueba.
- Entrenar el modelo.
- Generar predicciones.
- Evaluar y graficar resultados.

## Referencias

La carpeta [`PDF`](PDF/) incluye articulos y documentos utilizados como base teorica, entre ellos trabajos sobre Autoformer, Transformers para series temporales, Chronos y forecasting financiero.

## Trabajo futuro

- Separar el codigo del notebook en scripts reutilizables.
- Agregar un archivo `requirements.txt` o `environment.yml`.
- Corregir y estandarizar la nomenclatura de modelos en todas las graficas.
- Agregar validacion walk-forward para evaluar robustez temporal.
- Incluir backtesting y comparacion contra modelos baseline simples.
- Documentar experimentos Autoformer completos con checkpoints y configuraciones.

## Licencia

Este repositorio no incluye todavia una licencia explicita. Antes de reutilizarlo o distribuirlo, agrega una licencia adecuada para tu caso de uso.
