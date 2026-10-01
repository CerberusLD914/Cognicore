"""
setup_colab.py — Verifica la instalación de JAX y la GPU en Colab.

Ejecuta este script en Colab para verificar que todo esté correcto:
    !python setup_colab.py
"""

import sys


def check_gpu():
    """Verifica que la GPU esté disponible."""
    print("=" * 60)
    print("VERIFICACIÓN DE GPU")
    print("=" * 60)

    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("⚠️  nvidia-smi no disponible")
    except Exception as e:
        print(f"⚠️  Error ejecutando nvidia-smi: {e}")


def check_jax():
    """Verifica que JAX esté instalado con soporte GPU."""
    print("=" * 60)
    print("VERIFICACIÓN DE JAX")
    print("=" * 60)

    try:
        import jax
        import jax.numpy as jnp

        print(f"JAX version: {jax.__version__}")
        print(f"Backend: {jax.default_backend()}")
        print(f"Dispositivos: {jax.devices()}")

        # Test simple en GPU
        x = jnp.ones((1000, 1000))
        y = x @ x
        print(f"Test matmul en {y.device}: OK")

        return True
    except ImportError:
        print("❌ JAX no instalado. Ejecuta:")
        print("   !pip install -U 'jax[cuda12]' flax optax")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_flax():
    """Verifica que Flax esté instalado."""
    print("=" * 60)
    print("VERIFICACIÓN DE FLAX")
    print("=" * 60)

    try:
        import flax
        import optax
        print(f"Flax version: {flax.__version__}")
        print(f"Optax version: {optax.__version__}")
        return True
    except ImportError:
        print("❌ Flax/Optax no instalado. Ejecuta:")
        print("   !pip install flax optax")
        return False


def check_modules():
    """Verifica que los módulos de CogniCore-JAX estén accesibles."""
    print("=" * 60)
    print("VERIFICACIÓN DE MÓDULOS COGNICORE-JAX")
    print("=" * 60)

    try:
        from cognicore_jax.model import build
        from cognicore_jax.train import train
        from cognicore_jax.data import load_corpus

        print("✅ Todos los módulos importados correctamente")

        # Test de construcción del modelo
        model, cfg = build(target_params=1_000_000, seq_len=128)
        print(f"✅ Modelo de prueba construido: {cfg}")
        return True
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        print("   Asegúrate de que la carpeta cognicore_jax/ esté en el path")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def benchmark():
    """Benchmark rápido de velocidad GPU vs CPU."""
    print("=" * 60)
    print("BENCHMARK GPU")
    print("=" * 60)

    try:
        import jax
        import jax.numpy as jnp
        import time

        # Tamaño del benchmark
        n = 2048

        # GPU
        x = jnp.ones((n, n))
        y = jnp.ones((n, n))

        # Warmup
        for _ in range(3):
            z = x @ y
        z.block_until_ready()

        # Benchmark
        t0 = time.time()
        for _ in range(10):
            z = x @ y
        z.block_until_ready()
        gpu_time = (time.time() - t0) / 10

        print(f"Matmul {n}x{n} en GPU: {gpu_time*1000:.2f} ms")
        print(f"Throughput: {2*n**3/gpu_time/1e9:.1f} GFLOPS")

    except Exception as e:
        print(f"❌ Error en benchmark: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("COGNICORE-JAX: Verificación de entorno en Colab")
    print("=" * 60 + "\n")

    check_gpu()
    print()

    if check_jax():
        print()
        check_flax()
        print()
        check_modules()
        print()
        benchmark()
    else:
        print("\n❌ Instala JAX primero:")
        print("   !pip install -U 'jax[cuda12]' flax optax")

    print("\n" + "=" * 60)
    print("VERIFICACIÓN COMPLETA")
    print("=" * 60)
