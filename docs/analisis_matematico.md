# Matemática del cálculo y del análisis

Este documento describe, con la notación del código, todo lo que se
calcula: el modelo y el algoritmo que encuentra el campo de depinning y la
configuración crítica (`main.cu`, `misistema.h`), la extracción de la
pared (`scripts/wall_contour.py`), los estimadores de rugosidad
(`scripts/roughness_estimators.py`, `scripts/local_roughness.py`) y el
modelo de dos crossovers (`scripts/fit_crossovers.py`). Al final están los
resultados del barrido `L = 1024, 2048` (figuras en
`results/sweep_L1024_L2048/`).

Referencia: A. B. Kolton, E. E. Ferrero, A. Rosso, Phys. Rev. B **108**,
174201 (2023).

---

## 1. Modelo

Campo escalar $\phi(x,y)$ en una red $L\times L$:

```math
\partial_t \phi = c\,\nabla^2\phi + \epsilon_0\left[(1+r(x,y))\,\phi - \phi^3\right] + h
```

- $\nabla^2$ es el laplaciano discreto de 5 puntos.
- El desorden es de tipo *random bond*: $r(x,y)$ es independiente en
  cada sitio y uniforme en $[-\Delta,\Delta]$, así que
  $\langle r\rangle = 0$ y $\langle r^2\rangle = \Delta^2/3$
  (`site_disorder`, generador Random123 indexado por sitio y semilla).
- $c$ = `CEL`, $\epsilon_0$ = `EPSILON0` y $\Delta$ = `AMPDIS`; en las
  corridas, $c = \epsilon_0 = 1$.

**Bordes.** En $x$ son periódicos, $\phi(x+L,y) = \phi(x,y)$. En $y$ son
antiperiódicos, $\phi(x,y+L) = -\phi(x,y)$. Con borde antiperiódico,
cualquier configuración tiene un número impar de paredes, así que siempre
hay una pared que atraviesa la muestra. Para $h>0$, la región $\phi>0$
está debajo de la pared y crece al aumentar $h$.

**Estados de equilibrio.** Sin desorden ni campo, los mínimos locales son
$\phi = \pm 1$. La pared plana es $\phi_0(y-u) = \tanh[(y-u)/\xi\sqrt2]$
(con el signo del borde), de ancho $\xi = \sqrt{c/\epsilon_0}$ en el
continuo.

## 2. Variant Monte Carlo (VMC)

### 2.1 Movimiento elemental

En cada sitio, la condición estacionaria $\partial_t\phi_i = 0$ con los
vecinos fijos es una cúbica en $\phi_i$:

```math
c\left(\textstyle\sum_{\rm nn}\phi - 4\phi_i\right) + \epsilon_0\left[(1+r_i)\phi_i - \phi_i^3\right] + h = 0
\;\Longleftrightarrow\;
\phi_i^3 + p\,\phi_i - q = 0,
```

```math
p = \frac{4c}{\epsilon_0} - (1+r_i), \qquad q = \frac{c\sum_{\rm nn}\phi + h}{\epsilon_0}.
```

`vieta_solver` resuelve la cúbica. Con $Q = p/3$, $R = q/2$ y
$D = Q^3 + R^2$:

- **Si $D \ge 0$** hay una sola raíz real (Cardano):
  $\phi = \sqrt[3]{R+\sqrt D} + \sqrt[3]{R-\sqrt D}$.
- **Si $D < 0$** hay tres raíces reales (forma trigonométrica de Vieta):

```math
\phi_k = 2\sqrt{-Q}\,\cos\frac{\theta + 2\pi k}{3},\quad k=0,1,2,\qquad \theta = \arccos\frac{R}{\sqrt{-Q^3}},
```

con $\phi_0 > \phi_2 > \phi_1$: la mayor, la del medio y la menor.

La raíz del medio, $\phi_2$, es inestable. El movimiento lleva $\phi_i$ a
la raíz estable de la misma cuenca que el valor actual: a $\phi_0$ si
$\phi_i > \phi_2$, y a $\phi_1$ si no. Así, el sitio salta directamente al
mínimo local más cercano en lugar de integrar la dinámica.

### 2.2 Barrido y convergencia

- **Damero.** Los sitios con $x+y$ par se actualizan todos en paralelo y
  después los impares (`vmcop`, dos colores por barrido). Como los cuatro
  vecinos de un sitio son del otro color, cada media actualización es
  exacta.
- **Velocidad residual.** Cada `chop` = 100 barridos se calcula

```math
v = \frac{1}{L}\sum_{i=1}^{L^2}\partial_t\phi_i
```

  (`velocity_phi`). La normalización es por $L$, no por $L^2$: como el
  salto de $\phi$ a través de la pared es 2, una pared de largo $L$ que
  avanza con velocidad media $\bar v_{\rm pared}$ da
  $\sum_i\partial_t\phi_i\approx 2L\,\bar v_{\rm pared}$, así que
  $v\approx 2\bar v_{\rm pared}$, sin importar el tamaño.
- **Parada.** La relajación termina cuando $|v| <$ `TOLVEL` $= 10^{-4}$
  (estado metaestable), o al llegar a `MAXITERATIONS` $= 5000$ bloques
  ($5\times10^5$ barridos).

### 2.3 Campo de depinning por bisección

Para cada $h$ de prueba (`find_depinning_field`):

1. **Condición inicial plana:** $\phi = +1$ para $y < 0.2L$ y $\phi=-1$
   para $y \ge 0.2L$.
2. **Relajación** con la regla anterior, con salida anticipada: si la
   pared cruza la fila $y_e = 0.95L$, es decir si
   $\max_x \phi(x, y_e) > 0$ (`magnetization_lane`, `ESCAPE_LANE`), se
   corta ahí.
3. **Clasificación:** el intento es *depinned* si
   $\max_x\phi(x,y_e) > 0$ y *pinned* si no.

La bisección sobre $[a,b]$ reemplaza $a$ (pinned) o $b$ (depinned) por el
punto medio hasta que $b-a \le$ `tol` $= 10^{-4}$. Después retrocede,
$a \leftarrow a - \text{tol}$, hasta que el intento quede pinned. Así se
define

```math
h_d \equiv a, \qquad \text{configuración crítica} = \text{estado metaestable a } h_d.
```

**Por qué reiniciar desde la pared plana es válido.** Por los teoremas de
no-adelantamiento de Middleton, la dinámica preserva el orden: si
$\phi \le \phi'$ en todo sitio, sigue siéndolo. Por eso, a $h$ fijo, que
la pared quede trabada o escape no depende del camino. En concreto, la
regla VMC es monótona: las raíces de la cúbica crecen con $q$, es decir
con los vecinos y con $h$, y el umbral $\phi_2$ baja.

**Por qué la salida anticipada no cambia $h_d$.** Con borde
antiperiódico, la magnetización $M = \sum\phi$ crece mientras la pared
sube y es máxima con la pared en $y = L$. Una pared depinned avanza hasta
ahí y converge, en lugar de dar vueltas. Cortar al cruzar $y_e$ no cambia
la clasificación, porque la decisión ya se toma con esa fila. Los intentos
pinned nunca cruzan $y_e$, así que relajan igual que antes y la
configuración crítica guardada no cambia. Se verificó con 13 semillas: el
mismo $h_d$, y en $L=1024$ el tiempo total bajó un 32 %.

**Intervalo inicial.** $[0,\, 3h_{\rm est}]$, con
$h_{\rm est} = 0.008\,(\Delta/0.2)^{4/3}$ (escalamiento de Larkin, §7).

## 3. La pared como curva

La configuración crítica se guarda completa (`--dump-field`). Los puntos de
`critica_*.dat` no sirven para este análisis: solo detectan cambios de
signo a lo largo de $y$, así que pierden los tramos verticales (los
overhangs) y no vienen ordenados.

1. **Contorno.** La pared es la curva de nivel $\phi = 0$, trazada con
   *marching squares* con resolución subpíxel (`skimage.measure.find_contours`)
   sobre el campo repetido 3 veces en $x$. Se toma el contorno abierto más
   largo que atraviesa las tres copias sin tocar las filas del borde en
   $y$, y se corta exactamente un período, del punto $P$ a $P + (L,0)$.
   Los contornos cerrados son burbujas, dominios aislados: se cuentan y se
   descartan.
2. **Remuestreo por longitud de arco.** La curva se remuestrea con $n$
   puntos equiespaciados en longitud de arco $s$, con
   $ds = s_{\rm tot}/n \approx 1$ (`periodic_arclength`), de modo que el
   punto $n$ sea la imagen del punto 0. La longitud de arco total por
   período, $s_{\rm tot}/L \ge 1$, mide cuánto se aparta la pared de una
   recta.
3. **Perfil univaluado** (ec. 9 del paper, `univalued_profile`). En cada
   columna $x+\tfrac12$, la curva corta $m(x)$ veces (un número impar)
   en alturas $u_n(x)$, y se define

```math
u(x) = \frac{1}{m(x)}\sum_{n=1}^{m(x)} u_n(x).
```

   Una columna tiene overhang si $m(x)\ge 3$. A partir de esto,
   $\rho_{\rm cols} = \langle\, \mathbb 1[m(x)\ge3]\,\rangle_x$ es la
   fracción de columnas multivaluadas, y una muestra cuenta como "con
   overhang" si tiene al menos una.

## 4. Observables de rugosidad

### 4.1 Globales, sobre $u(x)$

Con $\delta u = u - \bar u$ y $q_k = 2\pi k/L$ ($k = 1..L/2$):

```math
S(q_k) = \frac{1}{L}\left|\sum_x \delta u(x)\,e^{-iq_kx}\right|^2,\qquad
B(r) = \langle[u(x+r)-u(x)]^2\rangle_x = 2[C(0)-C(r)],\qquad
W^2 = \langle\delta u^2\rangle_x,
```

donde $C(r)$ es la autocorrelación circular, calculada con FFT. Para una
interfaz autoafín con exponente $\zeta$:

```math
S(q)\sim q^{-(1+2\zeta)},\qquad B(r)\sim r^{2\zeta},\qquad W^2\sim L^{2\zeta}.
```

**Limitación para $\zeta>1$** (superrugosidad, como qEW con
$\zeta\approx1.25$). Como $|u(x+r)-u(x)| \le r\max|u'|$, $B(r)$ no puede
crecer más rápido que $r^2$: para $\zeta > 1$, los incrementos quedan
dominados por la pendiente local y $B \sim r^{2\min(\zeta,1)}$. Lo mismo
pasa con $W^2$ de toda la muestra. Por eso, $B$ y $W^2$ globales reportan
$\zeta = 1$, y entre los globales solo $S(q)$ ve $\zeta>1$.

### 4.2 Locales, sobre segmentos rotados

Se toman todos los segmentos de longitud de arco $l$ (con $l+1$ puntos,
uno cada $l/2$ puntos a lo largo del período, `_segments`). Cada segmento
se rota rígidamente para que su inclinación $\theta$ quede horizontal.
Con $(dx, dy)$ medidos desde el primer punto:

- **Cuerda** (`chord`): $\theta = \operatorname{atan2}(dy_{\rm fin}, dx_{\rm fin})$,
  la recta entre los extremos.
- **Eje principal** (`pca`): $\theta = \tfrac12\operatorname{atan2}(2s_{xy},\, s_{xx}-s_{yy})$,
  con $s$ las covarianzas de los puntos del segmento. Es el eje de mínima
  dispersión perpendicular.

```math
X_r(s) = \cos\theta\,dx + \sin\theta\,dy,\qquad h(s) = -\sin\theta\,dx + \cos\theta\,dy.
```

$h(s)$, la coordenada perpendicular al eje en función de la longitud de
arco $s$, está bien definida aunque la pared tenga overhangs. Promediando
sobre segmentos:

```math
W^2(l) = \left\langle \operatorname{var}_s h \right\rangle,\qquad
B(r,l) = \left\langle[h(s+r)-h(s)]^2\right\rangle_{s},\qquad
\theta^2(l) = \langle\theta^2\rangle .
```

En el régimen autoafín, $W^2(l)\sim l^{2\zeta}$ incluso para $\zeta>1$:
al restar la inclinación de cada segmento desaparece la componente
$\sim l^2$ que satura $B$. Los perfiles sintéticos con $\zeta = 1.25$
puro lo confirman: $\zeta_W$ lee entre 1.20 y 1.27 para
$7 \le l \lesssim L/6$, sin sesgo hacia arriba a escalas chicas. En
cambio, $B(r,l)$ sigue limitado a $\zeta\le1$, porque los incrementos
dentro del segmento también están dominados por la pendiente local; en
los datos lee ≈ 0.8.

**Factor de estructura local.** Con $\delta h = h - \bar h$:

```math
\text{cuerda (FFT):}\quad S(q,l) = \frac{ds}{l}\left|\sum_s \delta h(s)e^{-iqs}\right|^2,\quad q = \frac{2\pi k}{l};
```

```math
\text{eje principal (DCT-II):}\quad S(q,l) = \frac{ds}{2l}\,\big[\mathrm{DCT}(\delta h)_k\big]^2,\quad q = \frac{\pi k}{l}.
```

La DCT es la FFT de la extensión par de $h$, de período $2l$. **Por qué
cada transformada:** la FFT supone que el segmento es periódico.

- Si $h(0)\ne h(l)$, como pasa con el eje principal, el salto en el borde
  da coeficientes $\sim 1/q$ y $S\sim q^{-2}$, que equivale a un
  $\zeta_{\rm aparente} = 0.5$.
- Con la cuerda, $h(0)=h(l)=0$ por construcción, así que no hay salto.
- La extensión par tampoco tiene salto, solo un quiebre de pendiente, que
  da $\sim 1/q^2$ y $S\sim q^{-4}$, compatible con medir $\zeta < 1.5$.

### 4.3 Fracción multivaluada

Un segmento sigue siendo multivaluado en su propio marco si su coordenada
a lo largo del eje retrocede más de una unidad de red:

```math
f_{\rm mv}(l) = \Big\langle \mathbb 1\Big[\max_s\big(\max_{s'\le s}X_r(s') - X_r(s)\big) > 1\Big]\Big\rangle_{\rm segmentos}.
```

Vale $0$ para una pared autoafín (a escala $l$ es un grafo en su propio
marco) y tiende a $1$ cuando los overhangs dominan esa escala.

## 5. Exponentes locales y longitud de crossover

### 5.1 Exponente local (dependiente de la escala)

Para una curva $y(x)$ en escala log-log, la pendiente local se calcula
como la de la recta de cuadrados mínimos de $\log y$ contra $\log x$,
usando los puntos a menos de $\pm0.15$ décadas de cada $x$ (`local_slope`):

```math
\zeta_W(l) = \tfrac12\,\frac{d\ln W^2}{d\ln l},\qquad
\zeta_S^{\rm glob}(\ell = 2\pi/q) = -\tfrac12\left(\frac{d\ln S}{d\ln q}+1\right).
```

$S$ global se promedia antes en bins logarítmicos (12 por década). Desde
el espectro local se ajusta $S(q,l)\sim q^{-(1+2\zeta)}$ en $k=2..8$ (FFT)
o $k=4..16$ (DCT), es decir en longitudes de onda $l/8..l/2$ en los dos
casos, y desde $B(r,l)\sim r^{2\zeta}$ en $r\in[l/16,\, l/4]$.

### 5.2 Definiciones de $l_o$

$l_o(L,\Delta)$ es la primera escala, interpolada en $\ln l$, donde:

| definición | criterio |
|---|---|
| `zW_chord`, `zW_pca` | $\zeta_W(l)$ baja de $\zeta_{\rm th} = 1.125$ (para $l\ge 8$) |
| `zS_glob` | $\zeta_S^{\rm glob}$ baja de $\zeta_{\rm th}$ |
| `fmv_chord`, `fmv_pca` | $f_{\rm mv}(l)$ sube por encima de $0.05$ |

Se usa un umbral bajo para $f_{\rm mv}$ porque solo se acerca a 1 cuando
$L \gg l_o$.

### 5.3 Colapso de datos

A $L$ fijo, si $g(l) = G(l/l_o(\Delta))$ para una función $G$ común, al
desplazar las curvas en $\ln l$ por $a_\Delta = \ln[l_o(\Delta)/l_o(\Delta_0)]$
todas caen una sobre otra. `collapse_shifts` minimiza

```math
\mathcal C(\{a\}) = \frac{1}{N_{\rm pares}}\sum_{i<j}\big\langle [g_i(\ln l + a_i) - g_j(\ln l + a_j)]^2\big\rangle_{\rm solapamiento},
```

con $a_{\Delta_0} = 0$ (Nelder–Mead). Si dos curvas se solapan en menos de
0.3 en $\ln l$, el par se penaliza. Para $f_{\rm mv}$ solo se usa la parte
de cada curva con $f_{\rm mv}\ge0.01$, porque una curva idénticamente nula
colapsa con cualquier cosa.

### 5.4 Exponente $\alpha$ y corte de tamaño finito

```math
l_o \sim \Delta^{-\alpha}:\qquad \ln l_o = c - \alpha\ln\Delta \quad\text{(cuadrados mínimos)}.
```

El paper da $\alpha \approx 1/0.45 \approx 2.2$.

**Tamaño finito.** Aun sin crossover, $\zeta_W(l)$ cae cerca de
$l\sim L/2$, porque un segmento de longitud comparable a la muestra está
restringido por la periodicidad. Comparando los dos tamaños, $l_o$
coincide hasta $\approx L/4$ ($\Delta=0.7$: 240 y 237) y queda
sobreestimado más allá ($\Delta=0.6$: 365 contra 274, a $0.36L$). Por eso
(`--lo-max-frac 0.25`):

- los ajustes de $\alpha$ excluyen los $\Delta$ con $l_o > L/4$, decididos
  sobre la muestra completa;
- los colapsos excluyen los mismos $\Delta$ (según $l_o$ de
  $f_{\rm mv}$) y solo usan escalas $l\le L/4$.

### 5.5 Errores

Bootstrap sobre muestras: se remuestrean las $n$ muestras con reemplazo y
se recalculan todos los promedios, curvas, $l_o$, colapsos y ajustes. El
error de una cantidad es la desviación estándar entre remuestreos. Se usa
este método porque los puntos vecinos en $l$ están correlacionados, ya
que vienen de los mismos segmentos solapados, y entonces el $\chi^2$ de un
ajuste no da errores confiables.

## 6. Modelo de dos crossovers para $\zeta_W(l)$

`scripts/fit_crossovers.py`.

### 6.1 Exponente local

```math
\zeta(l) = \zeta_L + (\zeta_q-\zeta_L)\,\sigma\!\left(\frac{\ln(l/L_c)}{w_1}\right) + (\zeta_\infty-\zeta_q)\,\sigma\!\left(\frac{\ln(l/l_o)}{w_2}\right),
\qquad \sigma(x) = \frac{1}{1+e^{-x}},
```

```math
L_c = a\,\Delta^{-x_c},\qquad l_o = b\,\Delta^{-\alpha}.
```

Límites:

- $\zeta\to\zeta_L$ (Larkin) para $l\ll L_c$;
- $\zeta\approx\zeta_q$ (qEW) para $L_c\ll l\ll l_o$;
- $\zeta\to\zeta_\infty$ para $l\gg l_o$.

$L_c$ y $l_o$ son los centros de los crossovers; $w_1$ y $w_2$, sus
anchos en $\ln l$.

### 6.2 Forma equivalente para $W^2(l)$

Como $\zeta = \tfrac12\,d\ln W^2/d\ln l$, integrando se obtiene

```math
W^2(l) = A\,l^{2\zeta_L}\,\Big[1+(l/L_c)^{n_1}\Big]^{2(\zeta_q-\zeta_L)/n_1}\,\Big[1+(l/l_o)^{n_2}\Big]^{2(\zeta_\infty-\zeta_q)/n_2},\qquad n_i = 1/w_i,
```

porque

```math
\frac{d}{d\ln l}\ln\!\Big[1+(l/\ell)^n\Big] = \frac{(l/\ell)^n}{1+(l/\ell)^n} = \sigma\big(n\ln(l/\ell)\big).
```

Cada factor $[1+(l/\ell)^n]^{2\delta\zeta/n}$ cambia el exponente de $W^2$
en $2\delta\zeta$ alrededor de $l=\ell$, y $n$ mide qué tan abrupto es el
cambio. Es la forma estándar de un crossover entre dos leyes de potencia.

### 6.3 Por qué no se ve una meseta

Un escalón logístico pasa del 10 % al 90 % en
$\Delta\ln l = 2\ln 9\cdot w \approx 4.4\,w$, es decir
$\approx 1.9\,w$ décadas. Para que $\zeta(l)$ quede cerca de $\zeta_q$
hacen falta las dos colas saturadas, y para ver una meseta de ancho $D$
décadas,

```math
\log_{10}\frac{l_o}{L_c} \gtrsim 0.95\,(w_1 + w_2) + D .
```

Del ajuste: $l_o/L_c = (b/a)\,\Delta^{-(\alpha - x_c)} \approx 4\,\Delta^{-1.43}$.
Eso da ≈ 4 en $\Delta=1$ y ≈ 15 en $\Delta=0.4$, mientras que solo tocar
$\zeta_q$ ya requiere $\gtrsim 25$, y una década de meseta, $\gtrsim 250$
($\Delta\approx0.055$).

### 6.4 Ajuste

- **Datos:** todas las curvas $(L,\Delta)$ juntas, con
  $5\le l\le L/4$.
- **Pesos:** $\sigma_i$ es el error bootstrap de cada punto, con piso
  0.01.
- **Método:** cuadrados mínimos sobre $(\zeta_{\rm modelo}-\zeta_W)/\sigma$
  (`scipy.optimize.least_squares`), con $w\in[0.05,5]$,
  $x_c\in[0,5]$ y $\alpha\in[0,6]$.
- **Variantes:** todo libre; $x_c = 2/3$ fijo; $x_c$ y $\alpha=2.1$ fijos.
  $\zeta_L = 1.5$ fijo salvo con `--free-zeta-L`.
- **Errores:** por bootstrap, reajustando sobre las curvas remuestreadas.

**Colapso de Larkin.** La parte con $\zeta_W\ge1.3$ se colapsa por
desplazamientos en $\ln l$ (§5.3). Los desplazamientos se ajustan a
$\Delta^{-x_c}$.

## 7. Escalamiento de Larkin

**Elasticidad efectiva.** Para desplazamientos lentos, la pared es una
línea elástica $u(x)$ con tensión superficial $\gamma\sim\sqrt{c\,\epsilon_0}$.
El desorden *random bond* le acopla una energía

```math
V(x,u) = -\frac{\epsilon_0}{2}\int dy\; r(x,y)\,\phi_0^2(y-u).
```

Su correlador $R(u)$ tiene alcance $\xi$ en $u$ y amplitud
$\propto \epsilon_0^2\langle r^2\rangle\propto\Delta^2$.

**Teoría de perturbaciones (Larkin).** Una línea de largo $l$ se deforma
por la fuerza aleatoria acumulada:

```math
\overline{u^2}(l) \sim \frac{|R''(0)|}{\gamma^2}\,l^{\,4-d}\quad (d=1:\; l^3)
\;\Rightarrow\; \zeta_L = \frac{4-d}{2} = \frac32 .
```

**Longitud de Larkin.** Es la escala en que la deformación llega al ancho
de la pared, $u(L_c)\sim\xi$:

```math
L_c \sim \left(\frac{\gamma^2\,\xi^2}{|R''(0)|}\right)^{1/3}\propto\Delta^{-2/3},
\qquad |R''(0)|\sim R(0)/\xi^2\propto\Delta^2.
```

**Campo crítico.** La fuerza por unidad de longitud del campo sobre la
pared es $2h$ (el salto de $\phi$ es 2). Igualándola a la elástica en
$L_c$:

```math
2h_d \sim \frac{\gamma\,\xi}{L_c^2}\;\Rightarrow\; h_d\propto\Delta^{4/3}.
```

Por encima de $L_c$ empieza el régimen qEW ($\zeta\approx1.25$ en
$d=1$), hasta $l_o$.

## 8. Resultados ($L=1024$ y $2048$; $\Delta=0.4$–$1.0$; 30 muestras)

| cantidad | resultado |
|---|---|
| $h_d$ | $\propto\Delta^{1.343\pm0.007}$ ($L=1024$), $\Delta^{1.342\pm0.005}$ ($L=2048$); valores iguales entre $L$ al ≲ 0.8 % |
| $l_o$ con $f_{\rm mv}$, cuerda ($L=2048$) | 513, 274, 237, 136, 106 para $\Delta$ = 0.5, 0.6, 0.7, 0.85, 1.0; independiente de $L$ para $\Delta\ge0.7$ |
| $\alpha$ ($L=2048$, $l_o\le L/4$) | $f_{\rm mv}$: 1.98 ± 0.15 (cuerda), 2.20 ± 0.23 (eje principal); colapsos: 2.06 ± 0.12, 2.23 ± 0.14 |
| colapso de Larkin | $x_c = 0.65\pm0.06$ ($L=2048$), $0.83\pm0.09$ ($L=1024$); compatible con 2/3 |
| ajuste de dos crossovers | $\zeta_L$ libre: 1.47–1.51; $\zeta_q = 1.203\pm0.014$ con $x_c$, $\alpha$ fijos; entre 1.11 y 1.22 según la variante |
| $l_o$ con $\zeta_W$ o con $S$ global | no confiables ($\alpha$ entre −2.7 y 1.4): el umbral 1.125 cae donde se mezclan los dos crossovers |

**Lectura.**

- **Larkin, confirmado.** El ~1.45 de escalas chicas es físico: no
  aparece en perfiles sintéticos con $\zeta = 1.25$, escala con
  $\Delta^{-2/3}$ y el ajuste con $\zeta_L$ libre da 1.5.
- **$\zeta_q$, no resuelto.** El colapso de Larkin sigue funcionando por
  debajo de 1.25, hasta ≈ 1.2, y el ajuste da $\zeta_q\approx1.20$. El
  error sistemático por la forma del modelo es ~0.05–0.1, así que no hay
  una diferencia significativa con 1.25.
- **La ventana qEW es demasiado angosta.** Con $l_o/L_c\approx4$–15 no
  alcanza para resolverla; según §6.3, haría falta $\Delta$ mucho más chico.
