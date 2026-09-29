# CogniCore — HDS-LSM

**Un tipo de entrenamiento de IA que no es un LLM.** Sin atención, sin KV cache,
sin tabla de embeddings, sin tokenizer. Entrena en CPU con NumPy puro.

```
py run_all.py
```

Un comando: verifica los gradientes, descarga el dataset, construye el modelo
de 10M, entrena, evalúa y cuantiza.

---

## Por qué esto no es un LLM

Un LLM tiene cuatro propiedades estructurales. Este modelo no tiene ninguna:

| | LLM (GPT-2, Llama, Qwen…) | CogniCore |
|---|---|---|
| Mezcla de tokens | self-attention, matriz O(T²) | SSM selectivo + conv de 4 taps |
| Memoria por token | KV cache O(T), **obligatoria** | estado fijo O(1), para siempre |
| Representación | tabla `Embedding(vocab, d)` | hypervectores HDE, **0 parámetros** |
| Unidad de entrada | token (BPE, vocabulario cerrado) | **byte** (0–255) |

No es "un transformer con tweaks". La ruta de cálculo es distinta de principio a fin.

---

## Las cuatro piezas nuevas

### 1. HDE — Hyperdimensional Encoder (0 parámetros)

Todo LLM arranca con una tabla de embeddings. En GPT-2 small son 38M de los
124M parámetros, y **cierra el vocabulario**: el modelo solo puede emitir tokens
que existían al inicializar.

CogniCore lo reemplaza por un encoding hiperdimensional:

1. **Atom** — el byte siembra un hypervector en R^D generado al vuelo desde una
   base aleatoria con semilla fija (base de Kanerva, ±1 disperso).
2. **Bind** — convolución circular con hypervectores de *rol*.
3. **Bundle** — superposición por suma.

Los 5 roles: `content`, `depth` (nivel de anidamiento mod 8), `column`
(posición en la línea mod 16), `class` (8 clases de carácter), `tag`.

Consecuencias:
- **0 parámetros** de embedding
- **sin vocabulario fijo** — cualquier secuencia de bytes es representable
- la representación se regenera desde una semilla de 64 bits: **0 bytes** que almacenar
- el mismo HDE hace de un-embedding en la salida, así que la cabeza también encoge

La similitud coseno contra los 257 átomos **es** el un-binding: la cabeza
descodifica un hypervector superpuesto en lugar de consultar una tabla. Al ser
un solo GEMM contra una matriz constante, el gradiente fluye entero.

### 2. SSM selectivo con backward en O(T)

```
dt_t = softplus(β·(Δ_t + bias)) / β
a_t  = exp(A · dt_t)          A < 0, aprendido
h_t  = a_t · h_{t-1} + dt_t · u_t
y_t  = (C · h_t) · silu(z_t)
```

Forward en forma logarítmica acumulativa (estable numéricamente):

```
P_t = cumsum(A·dt)_t
y_t = exp(P_t) · cumsum(dt·u·exp(-P))_t · C · silu(z)
```

El backward es el punto clave. El gradiente del estado es una recurrencia
reversa lineal, y se resuelve con un **escaneo asociativo de Hillis-Steele**
en O(log T) pasadas vectorizadas:

```
dh_t = gh_t + a_{t+1} · dh_{t+1}
```

El monoid `(C₁,B₁)∘(C₂,B₂) = (C₁C₂, B₁ + C₁B₂)` con composición a saltos
dobles da la forma cerrada. Esto es **10× más rápido** que un bucle en Python
(861 ms vs 8509 ms por paso) y nunca desborda, porque solo multiplica
decay ≤ 1 — a diferencia del truco `exp(cumsum(log a))`.

### 3. LCC — Local Cognitive Cells (memoria hash)

Un scratchpad direccionable por contenido, 32 slots × 48 dims = **1.5 KB por
capa, fijo para siempre**:

```
draft = soft_read(base_slots)        gather top-2, pesos softmax
bank  = W_write^T · draft / norm      fase escritura  (1 GEMM)
out   = W_read  · bank                fase lectura     (1 GEMM)
```

Solo **un** tensor aprendido por capa (la proyección de direcciones, d×M ≈ 18k).
El banco se inicializa desde una base hiperdimensional con semilla, igual que
el HDE. Las direcciones se tratan como discretas, los valores sí son
diferenciables — la convención estándar para memorias hash.

Aquí está la diferencia de fondo: un LLM **debe** llevar KV cache para
recordar cualquier cosa. La memoria de CogniCore son 1.5 KB y no crece.

### 4. Sustituto de atención

Atención es lo que hace que un LLM sea un LLM. Aquí se reemplaza por:

- **vecino de bytes de radio fijo** (±2), codificado con el mismo HDE
- **convolución depthwise causal** de 4 taps (k·C parámetros)

En ningún punto de la red se forma una matriz T×T.

### 5. Refinamiento iterativo

Segunda pasada: el byte se vuelve a leer a través de su ventana local y el
logit se corrige. `logits = pasada1 + pasada2`. Es corrección iterativa sobre
bytes, no muestreo de un vocabulario.

---

## El modelo

```
bytes → HDE (0 params) → proj_in → [ LCC block ] × N → RMSNorm
                                                             ↓
                                            head_read (d → D_hd) → un-bind
                                                             ↓
                                          257 logits de byte
                                        + pase de refinamiento
```

Cada bloque:

```
RMSNorm → conv depthwise k=4 → SSM selectivo → +memoria hash → +residual
         → RMSNorm → SwiGLU → +residual
```

**El presupuesto de parámetros es una restricción real y automática.**
`Config._solve()` busca `(d_model, n_layers)` para caer tan cerca de 10M como
sea posible, con penalización por profundidad degenerada:

```
10,000,000 params →  d_model=288, n_layers=8, d_ff=768   (error 0.21%)
                     fijo: 1,253,921   por capa: 1,105,056 × 8
                     HDE: 0 parámetros
```

---

## Verificación

Un grafo de gradientes escrito a mano está roto hasta que se demuestra lo
contrario. `gradcheck.py` compara contra una **referencia float64 independiente**
con perturbaciones numéricas:

```
[1b] selective_ssm vs brute-force float64 reference
  forward rel_err = 1.70e-08
  [OK ] d/du      rel_err=1.66e-07      [OK ] d/dA     rel_err=1.98e-07
  [OK ] d/ddelta  rel_err=2.44e-07      [OK ] d/dC     rel_err=1.72e-07
  [OK ] d/dz      rel_err=2.73e-07      [OK ] d/ddb    rel_err=2.19e-07
                                          [OK ] d/dds   rel_err=3.27e-07
[3] end-to-end: 0 parámetros sin gradiente, 0 NaN
```

`run_all.py` **se niega a entrenar** si estas pruebas fallan.

Los errores que encontró (todos reales, todos silenciosos si no se verifican):
- el forward del SSM no aplicaba el escalado `dt` al término de entrada
- `d dt/d delta = sigmoid(x)`, no `sigmoid(x)·β` — el `1/beta` se cancelaba
- `d dt/d scale = sigmoid(x)·d_raw − dt`
- el gradiente de RMSNorm le faltaba el factor `1/n`
- el backward indexaba el eje de batch en vez del de tiempo
- el head cortaba el grafo en `.data` → 90 parámetros sin gradiente

---

## Uso

```bash
entrenar.cmd                        # continuo, logs en vivo, Ctrl+C para parar
test rapido.cmd                     # 5 min, demuestra que funciona
chat.cmd                            # consola interactiva
run_all.bat                         # 10M, 1500 pasos, eval + cuantiza
```

O directo en Python:

```bash
py run_all.py                          # todo automático
py run_all.py --params 10000000 --steps 2000
py run_all.py --resume                 # continuar desde checkpoint
py run_all.py --no-network             # corpus sintético offline
py gradcheck.py                        # solo verificación
py smoke.py                            # prueba de 2 minutos
```

### Entrenamiento continuo con logs en vivo

```
entrenar.cmd
```

Corre **hasta que pulses Ctrl+C**, mostrando la tabla completa cada 25 pasos:

```
  step    train    bpb    val    best        lr      |g|   kB/s  MB seen  status
  ------------------------------------------------------------------------------
     25   4.8213  6.955       -  5.5491   3.00e-03    1.82    1.9      0.05  improving
     50   3.1024  4.475       -  5.5491   3.00e-03    1.55    1.9      0.10  improving
   500   1.2044  1.737   1.9821   1.9821   2.98e-03    0.94    1.9      1.02  improving  <- new best, saved
          progress ████████████░░░░░░░░░░ 500/20000 steps   elapsed 2m11s  ~1h26m left
```

Qué significan las columnas:

| columna | qué te dice |
|---|---|
| `train` | pérdida de entrenamiento (nats/byte) |
| `bpb` | **bits per byte** — el número clave. 8.0 = adivina al azar, 0 = perfecto |
| `val` | pérdida en datos que el modelo **no** ha visto (cada 500 pasos) |
| `best` | mejor validación alcanzada — es por eso que se guarda el checkpoint |
| `lr` | learning rate actual |
| `\|g\|` | norma del gradiente. Si se dispara, el entrenamiento se está desbordando |
| `kB/s` | velocidad |
| `MB seen` | bytes de entrenamiento procesados hasta ahora |

La columna `status` te dice directamente si conviene seguir:

- `improving` — la loss sigue bajando, déjalo correr
- `flat` — se estancó, la LR ya no aporta
- `RISING` — la loss está subiendo, algo va mal
- `<- new best, saved` — acaba de mejorar la validación y guardó pesos
- `<- no val gain for a while` — lleva rato sin mejorar en validación
- `[overfitting]` — `val` está por encima de `train`: está memorizando

Al parar con Ctrl+C guarda el modelo y muestra el historial de validación.

Opciones:
```bash
py live.py --params 50000000 --lr 0.002   # modelo de 50M
py live.py --every 10                     # log más seguido
py live.py --eval-every 200               # valida más a menudo
py live.py --no-network                   # corpus sintético
```

Todo se anexa a `checkpoints/<nombre>.log` para comparar Runs después.

### Dataset — automático, con cadena de respaldo

1. **HuggingFace** (`codeparrot/github-code-clean`, `the-stack-smol`) — por
   streaming, sin descargar archivos grandes enteros
2. **GitHub raw** — CPython stdlib, requests, numpy, flask, fastapi, scipy
3. **Sintético** — gramática de programas con estructura de largo alcance real
   (anidamiento, reutilización de variables entre scopes)

Si no hay red, el pipeline **no se rompe**: usa el corpus sintético.

Como el modelo trabaja en bytes crudos, no hay tokenizador, ni BPE, ni archivo
de vocabulario. `seq_len=256` significa 256 *bytes* de contexto real.

### Muestreo — sin KV cache

```python
from cognicore import generate as G
out = G.sample(model, prompt="def add(a, b):\n", n=500, temperature=0.8)
bpb = G.perplexity(model, held_out_text)   # bits por byte
```

Generar 100 KB cuesta la **misma memoria** que generar 100 bytes. La KV cache de
un LLM habría crecido hasta 100k × 24 × 2 × d floats.

### Cuantización int8

```python
from cognicore.quant import quantize
quantize(model, bits=8)   # 40.1 MB → 10.1 MB  (3.97x)
```

Fácil aquí y difícil en un transformer: el bucle caliente es una **recurrencia**
(acumulación elemento a elemento sobre estado fijo), que mapea a aritmética
entera de forma exacta. El estado puede quedarse en int8 todo el tiempo; solo
la lectura HDE toca float32. En un transformer, LayerNorm, softmax y la KV
cache quieren floats, así que casi siempre terminas en fp16 sin ganar mucho.

---

## Requisitos

```
python 3.10+
numpy >= 1.24
```

Eso es todo. Sin PyTorch, sin TensorFlow, sin CUDA — unas 400 líneas de autodiff
propio. Un modelo de 10M usa ~160 MB entre pesos, gradientes y estado de Adam.

## Estructura

```
cognicore/
  autograd.py   motor de autodiff en NumPy puro (~470 líneas)
  ssm.py        SSM selectivo + escaneo de Hillis-Steele
  hde.py        Hyperdimensional Encoder
  memory.py     Local Cognitive Cells
  model.py      Config con auto-solver de presupuesto + CogniCore
  data.py       descarga de dataset con 3 fuentes y respaldo sintético
  train.py      bucle de entrenamiento
  generate.py   muestreo byte a byte
  quant.py      cuantización post-entrenamiento
live.py         entrenamiento continuo con monitor en vivo
chat.py         consola interactiva
gradcheck.py    verificación numérica float64
quicktest.py    prueba rápida de 5 minutos
run_all.py      orquestador de un comando
```

### Lanzadores Windows

| archivo | acción |
|---|---|
| `entrenar.cmd` | entrenamiento continuo, logs en vivo, Ctrl+C para parar |
| `test rapido.cmd` | prueba de 5 minutos (2M params) |
| `chat.cmd` | consola interactiva con el modelo |
| `run_all.bat` | entrenamiento completo 10M + eval + cuantización |

Los `.cmd` están en **ASCII puro a propósito**: `cmd.exe` los lee con el
codepage ANSI del sistema, y los acentos en un batch UTF-8 rompen el parser
en silencio (fue un bug real durante el desarrollo).

## Honestidad sobre los límites

- Un modelo de 10M no va a escribir producción. La calidad real necesita miles
  de millones de tokens; aquí se demuestra que la **receta** funciona y es
  entrenable en un portátil.
- El recurrente SSM es de la familia Mamba/RWKV. La parte genuinamente nueva
  aquí es la combinación: HDE sin embeddings + memoria hash de tamaño fijo +
  cuantización entera exacta, lo que da un modelo entrenable en NumPy puro.
- La velocidad viene de la estructura (O(T), sin O(T²)), no de multiprecisión.
  861 ms por paso de 4×128 en CPU; en una GPU sería marginalmente más rápido
  y mucho más rápido con threads de BLAS.
