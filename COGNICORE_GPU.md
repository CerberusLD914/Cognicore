# CogniCore-JAX: Entrenamiento en GPU con Google Colab

Este directorio contiene la versión JAX de CogniCore para entrenamiento acelerado por GPU en Google Colab.

## Estructura

```
cognicore_jax/
├── __init__.py
├── hde.py          # Hyperdimensional Encoder (0 parámetros)
├── ssm.py          # Selective State-Space scan con jax.lax.scan
├── memory.py       # LCC: Local Cognitive Cells (memoria hash)
├── model.py        # Modelo CogniCore en Flax
├── train.py        # Bucle de entrenamiento con JIT
└── data.py         # Pipeline de datos

CogniCore_GPU.ipynb  # Notebook de Colab listo para ejecutar
```

## Inicio rápido en Google Colab

### Paso 1: Preparar los archivos

**Opción A — Subir a Google Drive:**
1. Sube la carpeta `cognicore_jax/` a tu Google Drive
2. Sube `Cognicore_GPU.ipynb` a tu Google Drive
3. Abre el notebook desde Google Drive (clic derecho > Abrir con > Colab)

**Opción B — Subir directamente a Colab:**
1. Abre https://colab.research.google.com
2. File > Upload notebook > selecciona `CogniCore_GPU.ipynb`
3. Sube la carpeta `cognicore_jax/` usando el panel de archivos

### Paso 2: Configurar GPU

1. **Runtime > Change runtime type**
2. **Hardware accelerator: GPU** (T4 en versión gratuita, A100 en Pro)
3. **Save**

### Paso 3: Ejecutar

Ejecuta las celdas en orden:
1. Verificar GPU
2. Instalar JAX con CUDA
3. Montar Google Drive
4. Entrenar
5. Evaluar y generar

## Aceleración esperada

| Dispositivo | Velocidad | VRAM | Costo |
|-------------|-----------|------|-------|
| CPU (NumPy original) | 1x | ~500 MB RAM | Gratis |
| GPU T4 (Colab gratuito) | ~10-20x | 15 GB | Gratis |
| GPU A100 (Colab Pro) | ~30-50x | 40 GB | ~$50/mes |
| GPU V100 (Colab Pro) | ~20-30x | 16 GB | ~$50/mes |

## Optimizaciones implementadas

| Optimización | Descripción |
|--------------|-------------|
| `jax.jit` | Compila forward+backward en un solo kernel GPU |
| `jax.grad` | Diferenciación automática (sin backward manual) |
| `jax.lax.scan` | Scan recurrente eficiente para SSM |
| `optax.adamw` | AdamW con decoupled weight decay |
| `optax.clip_by_global_norm` | Gradient clipping |
| Flax | Definición modular del modelo |

## Configuración recomendada

### Para Colab gratuito (T4)
```python
TARGET_PARAMS = 10_000_000  # 10M parámetros
CORPUS_MB = 8
STEPS = 2000
BATCH = 8
SEQ_LEN = 256
LR = 3e-3
```

### Para Colab Pro (A100)
```python
TARGET_PARAMS = 50_000_000  # 50M parámetros
CORPUS_MB = 32
STEPS = 5000
BATCH = 16
SEQ_LEN = 512
LR = 3e-3
```

## Notas importantes

1. **Primera ejecución**: La compilación JIT puede tardar 1-2 minutos. Es normal.
2. **VRAM**: Un modelo 10M usa ~200 MB de VRAM (pesos + gradientes + optimizer states).
3. **Checkpoints**: Se guardan en Google Drive automáticamente.
4. **Reanudación**: El código soporta `resume=True` para continuar desde un checkpoint.

## Comparación con la versión original

| Característica | Original (NumPy) | JAX (GPU) |
|----------------|------------------|-----------|
| Framework | NumPy puro | JAX + Flax |
| Autodiff | Manual (~400 líneas) | Automático (jax.grad) |
| GPU | No | Sí |
| Velocidad 10M | ~500 kB/s | ~5000-10000 kB/s |
| Memoria | ~500 MB RAM | ~200 MB VRAM |
| Dependencias | numpy | jax, flax, optax |

## Solución de problemas

### "No GPU available"
- Ve a Runtime > Change runtime type > GPU
- Reinicia el runtime: Runtime > Restart runtime

### "Out of memory"
- Reduce `BATCH` o `SEQ_LEN`
- Reduce `TARGET_PARAMS`

### "JAX no detecta GPU"
```python
# Verificar instalación
!pip install -U "jax[cuda12]"
import jax
print(jax.devices())  # Debe mostrar GPU
```

### "La compilación JIT es lenta"
- Es normal en la primera ejecución (1-2 min)
- Las ejecuciones posteriores son instantáneas
- Usa `jax.clear_caches()` si cambias el modelo

## Código de ejemplo

```python
from cognicore_jax.train import train

# Entrenar 10M parámetros en GPU
state, model, cfg, hist, best = train(
    target_params=10_000_000,
    corpus_bytes=8_000_000,
    steps=2000,
    batch=8,
    seq_len=256,
    lr=3e-3,
    checkpoint_dir="checkpoints"
)

print(f"Mejor val loss: {best:.4f}")
print(f"Bits/byte: {best / np.log(2):.4f}")
```

## Licencia

Mismo licenciamiento que CogniCore original.
