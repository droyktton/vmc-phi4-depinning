# Caso anisótropo completo con desorden débil: ¿aparece la clase qKPZ de depinning?

## Contexto
**Qué ya sabemos.** Con desorden en bloques AX×AY alargados a lo largo de la pared (4×1, 8×1), h_d(s) se curva
hacia abajo (λ > 0). Eso apunta a qKPZ (percolación dirigida en 1+1). Los datos son de L = 512, Δ = 0.3 y en
caja isótropa (§5.8 de `docs/analysis_math.md`).

**Qué falta.** Verlo en las observables de escala, a desorden débil, para que el régimen elástico (sin overhangs)
sea largo:
- S(q);
- W² local;
- σ(h_d) vs L;
- h_d vs tilt;
- con ⟨h_d⟩ fijo al variar L (regla κ = P/w = 20).

**Predicciones.**

| | ζ | S(q) ~ q^-(1+2ζ) | 1/ν |
|---|---|---|---|
| qKPZ | ≈ 0.63 | q^-2.26 | ≈ 0.58 |
| qEW (lo que medimos en el caso isótropo) | 1.25 | — | ≈ 0.75 |

**Argumento de escalas** (para elegir Δ y bloque):
- El término KPZ pesa cuando la pendiente local s(l) ~ w(l)/l alcanza s* ~ |h''/h|^(−1/2). Eso da ≈ 0.66 para
  4×1 y ≈ 0.4 para 8×1.
- Los overhangs aparecen recién con s(l) ~ 1. Con ζ = 1.25, s ∝ l^0.25, así que l_KPZ ~ s*⁴ · l_o: entre
  l_o/5 y l_o/40.
- Si aparece qKPZ, ζ < 1 y la pendiente local baja con l. Entonces la fracción multivaluada debería dejar de
  crecer: es un chequeo independiente.
- El desorden débil agranda l_o (isótropo: 102 Δ^−2.1), y así queda espacio para el cruce. Hay que verificar
  que h''/h no caiga con Δ.

## Decisiones tomadas
- **El anisoL viejo** (Δ = 0.3, caja isótropa) **se reemplaza.** Hay que cambiar el monitor bqvzbi2ho para que,
  cuando vuelva `/home`, lance el piloto (etapa 1) y no `data/aniso_L/run.sh`.
- **Forma del bloque:** se decide con el piloto.

## Etapa 1 — Piloto (L = 1024, s = 0)
- **Combinaciones:** bloques {4×1, 8×1} × Δ ∈ {0.1, 0.15, 0.2}, con 10 semillas y `--dump-field`.
- **Binarios:** `make AMPDIS=Δ AX=.. AY=1 TARGET=phi4vmc_new_D{Δ}_ax{AX}ay1`. Compilar en el frontend está
  permitido; es compilación, no cómputo.
- **Caja:** P inicial grande (κ holgado), con `--adapt-headroom`.
- **Qué medir:**
  - ⟨h_d⟩;
  - w;
  - fracción multivaluada y l_o;
  - ζ local (`scripts/local_roughness.py`).
- **Tilt corto:** s ∈ {0, ±0.1, ±0.2}, a L = 512, 10 semillas, para tener h''/h a ese Δ (`scripts/hd_tilt.py`).
- **Criterio de elección:** gana el caso con l_KPZ estimado más chico frente a l_o y frente al costo. Te muestro
  la tabla y elegimos juntos antes de la producción.

## Etapa 2 — Calibrar la caja (κ = 20) para el caso elegido
- **Medir w(L)** en L = 256, 512, 1024 y 2048, con 10 semillas, usando la misma definición de w que la del
  estudio κ (`data/kappa/`).
- **Fijar P(L) = 20·w(L).** w(L) no se extrapola con L^1.25: se ajusta localmente, porque el exponente puede
  cambiar.
- **Chequeo:** con esas cajas, ⟨h_d⟩ debe ser independiente de L dentro del error, como en el caso isótropo
  (≲ 0.8 %). Si deriva, reajustar P.

## Etapa 3 — Producción
- **h_d, σ(h_d) y campos:** L = 256, 512, 1024, 2048, 4096 (y 8192 si el costo lo permite, en la A10), a s = 0.
  - Semillas: 50 en los L chicos, porque el error de σ con N semillas es ≈ 1/√(2N); 30 en los grandes.
  - `--tol` ≤ σ/5. Hoy σ/⟨h_d⟩ es ~0.3 % a L grande, así que puede hacer falta `--tol 5e-6`.
  - `--dump-field`.
  - Script `data/aniso_weak/run.sh` y `tasks.txt`, con el mismo formato que `data/aniso_L/`.
- **h_d vs tilt:** L = 512, 1024 y 2048, con s ∈ {0, ±0.05, ±0.1, ±0.2, ±0.3}, 30 semillas y P de la etapa 2.
  Mismo esquema que `data/tilt_aniso/run.sh`.

## Análisis (cola cpu, nunca en el frontend)
1. **σ(h_d) vs L.** Extender `scripts/hd_variance.py` para que lea los nuevos directorios.
   - Mantener la maquinaria actual: corrección de Sheppard, bootstrap, χ² con pendiente fija 0.75 contra 0.577,
     pendiente libre.
   - Agregar: pendiente local entre L sucesivos y la curva isótropa de referencia.
2. **W² local y S(q,l) local.** Usar `scripts/local_roughness.py`, con los estimadores de cuerda y pca y su
   corrección de sesgo (§5.6). Lo que se busca:
   - ζ_W(l) yendo de 1.25 a ≈ 0.63 por encima de l_KPZ;
   - fracción multivaluada (crossover indicators).
3. **S(q) global.** Usar `scripts/structure_factor.py` sobre u(x). Si qKPZ aplana la pared, debería servir aquí:
   en el isótropo no sirvió porque 93 % de las paredes tenían overhangs. Comparar con q^−2.26 y q^−3.5.
4. **h_d(s).** Usar `scripts/hd_tilt.py`: curvatura λ vs L y forma de F_c(s). También ver si λ crece o se
   satura con L.
5. **⟨h_d⟩ vs L**, como control de la regla κ.
6. **Documentar** en `docs/analysis_math.md`, en una §5.9 nueva, con figuras en `results/aniso_weak/`, y
   commit.

## Archivos
- **Nuevos:** `data/aniso_weak/{pilot,calib,prod,tilt}/run.sh` y `tasks.txt`, que no se versionan.
- **A extender:**
  - `scripts/hd_variance.py`: lectura de directorios con bloques y pendiente local;
  - `scripts/local_roughness.py` y `scripts/structure_factor.py`, si hace falta, para que lean los directorios
    `ax*ay*`.

## Verificación
- **Regresión:** el piloto 4×1 a Δ = 0.3 (1 semilla) reproduce el h_d de §5.8, ≈ 0.0263 dentro de la
  dispersión.
- **Análisis:** `hd_variance.py` reproduce el 1/ν isótropo conocido (0.88 ± 0.03) antes de interpretar los
  datos nuevos.
- **Regla κ:** ⟨h_d⟩ plano en L antes de leer σ(h_d).
