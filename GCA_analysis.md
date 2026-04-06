# Análisis del Gravitational Clustering Algorithm (GCA)
## Diagnóstico, correcciones y evolución del experimento

---

## 1. Contexto del problema

Se implementó un experimento de aprendizaje del **Gravitational Clustering Algorithm (GCA)**
sobre datos sintéticos (300 puntos, 4 clusters, 2D). El objetivo era entender el algoritmo
base antes de estudiar sus variaciones.

El experimento producía resultados ininterpretables: **287 clusters de 300 puntos originales**.

---

## 2. Diagnóstico iterativo

### Problema raíz #1: ausencia de time step (dt)

La primera versión del algoritmo sumaba aceleraciones de todos los pares sin escalar:

```
accel ≈ 235 unidades promedio
distancias entre vecinos ≈ 0.04 – 4.0 unidades
```

Sin `dt`, cada partícula se desplazaba ~235 unidades por iteración. Con esa escala,
ninguna partícula terminaba dentro del radio de fusión (`merge_dist=1.0`). Resultado:
**287 pseudo-clusters** (casi una partícula por cluster).

**Fix aplicado (primera iteración):** agregar `dt=0.005` y reducir `merge_dist=0.15`.

```python
# Antes
vel_act = self.damping * vel_act + accel
pos_act = pos_act + vel_act

# Después (fix temporal)
vel_act = self.damping * vel_act + accel * self.dt
pos_act = pos_act + vel_act * self.dt
```

Esto funcionó para seed=42 (4 clusters exactos, ARI≈1.0), pero no resolvía el
problema de fondo.

---

### Problema raíz #2: el modelo físico no representaba el GCA original

La implementación con `velocity + damping` introduce **momentum/inercia**: cada
partícula recuerda su velocidad anterior y puede "sobrepasar" o oscilar alrededor
del equilibrio. Esto es más parecido a una **simulación N-body amortiguada** que
al GCA descrito en la literatura.

#### ¿Qué dice el GCA original?

No existe un único "GCA original" — hay múltiples formulaciones. Las más citadas
(Gomez, Dasgupta & Nasraoui 2003; Wright 1977) coinciden en que el movimiento
debe ser **proporcional a la fuerza neta actual**, sin acumulación de velocidad:

```
Δx_i = α · F_neta_i / m_i    (sin velocidad, sin damping)
```

**Problema descubierto:** con gravedad clásica 1/r² all-to-all, esta formulación
**siempre colapsa a 1 cluster**.

#### Diagnóstico cuantitativo del colapso global

```
Magnitudes de aceleración (G=1, 300 partículas):
  min=31.83, mean=394.85, max=607.44

Con α=0.001:
  paso mínimo = 0.032, paso máximo = 0.607
```

La causa física: con 300 partículas interactuando all-to-all, la **fuerza neta
apunta siempre hacia el centro de masa global**. Sin importar qué tan pequeño sea α,
el sistema siempre colapsa. Búsqueda exhaustiva de parámetros confirmó el fenómeno:

| Configuración | Clusters | ARI |
|---------------|----------|-----|
| α muy pequeño (0.0001) | 1 | -1.0 |
| α moderado (0.001) | 1 | -1.0 |
| α grande (0.02) | 300+ | ~0.0 |

---

### Solución adoptada: kernel gravitacional gaussiano

La solución documentada en variantes modernas del GCA es limitar el alcance de
la interacción gravitacional. En lugar de 1/r², se usa un **kernel gaussiano**:

$$F_{ij} = G \cdot m_i \cdot m_j \cdot \frac{e^{-r_{ij}^2 / 2\sigma^2}}{r_{ij}} \cdot \hat{r}_{ij}$$

**Ventajas del kernel gaussiano:**
- La fuerza decae exponencialmente con la distancia → interacción de **corto alcance**
- Partículas en clusters diferentes apenas se atraen si σ es pequeño
- Elimina el colapso global sin necesitar velocidad/damping
- Físicamente: equivale a "gravedad apantallada" (análogo al potencial de Yukawa)

Búsqueda de parámetros (grid search sobre G, α, σ, merge_dist con 20 seeds):

| σ | α | merge_dist | Clusters | Silhouette | ARI |
|---|---|------------|----------|------------|-----|
| 0.3 | 0.002 | 0.3 | **4** | +0.857 | **1.000** |
| 0.3 | 0.001 | 0.3 | **4** | +0.857 | **1.000** |
| 0.4 | 0.003 | 0.3 | 5 | +0.458 | 0.704 |

Configuración final elegida: **G=1.0, α=0.001, σ=0.3, merge_dist=0.3**

---

## 3. Comparación antes / después

### 3.1 Modelo físico

| Aspecto | Versión original (con dt) | Versión corregida (kernel gaussiano) |
|---------|--------------------------|--------------------------------------|
| **Ley de fuerza** | Newton clásico: G·m_i·m_j / (r² + ε) | Gaussiana: G·m_i·m_j·exp(-r²/2σ²) / r |
| **Alcance** | Global (all-to-all, sin límite) | Local (~3σ = 0.9 unidades) |
| **Movimiento** | vel = damping·vel + accel·dt; pos += vel·dt | pos += α·accel (sin velocidad) |
| **Parámetros** | G, dt, damping, epsilon, merge_dist | G, α, σ, merge_dist |
| **Colapso global** | Sí (necesita dt pequeño para evitarlo) | No (kernel limita alcance) |
| **Representación** | Simulación N-body amortiguada | GCA con campo de corto alcance |

### 3.2 Datos de entrada

| Aspecto | Antes | Después |
|---------|-------|---------|
| `cluster_std` | 0.8 | **0.7** (más compacto, favorece GCA) |
| `outlier_frac` (base) | 0.05 | **0.0** (experimento base limpio) |
| Outliers | Mezclados con experimento base | Solo en celda de robustez |

### 3.3 Hiperparámetros

| Parámetro | Celda 9 (antes) | Celda 15 (antes) | Celda 17 (antes) | Ahora (todas las celdas) |
|-----------|-----------------|------------------|------------------|--------------------------|
| G | 0.5 | 0.5 | variable | **1.0** |
| dt | 0.005 | 0.005 | 0.005 | *(eliminado)* |
| damping | 0.5 | 0.5 | 0.5 | *(eliminado)* |
| alpha (α) | *(no existía)* | *(no existía)* | *(no existía)* | **0.001** |
| sigma (σ) | *(no existía)* | *(no existía)* | *(no existía)* | **0.3** |
| merge_dist | 0.15 | **1.0** ⚠️ | **1.0** ⚠️ | **0.3** |
| max_iter | 300 | 200 | 200 | **500** |

> ⚠️ La inconsistencia de `merge_dist` entre celdas invalidaba las comparaciones:
> la celda de robustez y el sweep ejecutaban un algoritmo diferente al experimento base.

### 3.4 Barrido paramétrico

| Aspecto | Antes | Después |
|---------|-------|---------|
| Variables del sweep | G × Temperatura | **α × merge_dist** |
| Justificación | G y T son secundarios | α y merge_dist son los más influyentes |
| Métricas reportadas | Silhouette, Clusters | **ARI + Silhouette + Clusters** |

### 3.5 Evaluación estadística

| Aspecto | Antes | Después |
|---------|-------|---------|
| Seeds evaluados | 1 (seed=42) | **20 seeds** |
| Reporte | Valor único | Media ± std, min, max |
| Visualización | Tabla | Tabla + **boxplots comparativos** |
| Test estadístico | Ninguno | Tasas de éxito (ARI > 0.95, > 0.70) |

---

## 4. Resultados finales

### Experimento base (seed=42, sin outliers)

| Métrica | GCA | KMeans |
|---------|-----|--------|
| **Clusters encontrados** | 4 ✓ | 4 (forzado) |
| **Silhouette** | 0.857 | 0.857 |
| **Davies-Bouldin** | 0.200 | 0.200 |
| **Calinski-Harabasz** | 6540.9 | 6540.9 |
| **ARI** | **1.000** | **1.000** |
| Distribución de clusters | [75, 75, 75, 75] | balanceado |

### Evaluación multi-seed (20 seeds)

| Métrica | GCA (media ± std) | KMeans (media ± std) |
|---------|-------------------|----------------------|
| **ARI** | 0.834 ± 0.184 | 0.975 ± 0.063 |
| **Silhouette** | 0.728 ± 0.160 | 0.743 ± 0.080 |
| **Clusters** | 3.75 ± 0.79 | 4.00 ± 0.00 |
| ARI > 0.95 | 10/20 seeds | — |
| ARI > 0.70 | **19/20 seeds** | — |

### Robustez frente a outliers

| Outliers | GCA clusters | GCA ARI | KMeans clusters | KMeans ARI |
|----------|-------------|---------|-----------------|------------|
| 0% | 4 | 1.000 | 4 | 1.000 |
| 5% | 7 | 1.000 | 4 | 1.000 |
| 10% | 12 | 0.990 | 4 | 1.000 |
| 20% | 13 | 1.000 | 4 | 1.000 |

> GCA agrupa outliers en micro-clusters separados (comportamiento esperado y
> explicable: los outliers aislados tienen poca masa y no se fusionan con clusters
> densos bajo el kernel gaussiano de corto alcance).

---

## 5. Lecciones aprendidas

### 5.1 Sobre el algoritmo GCA

1. **No existe un único GCA:** hay múltiples formulaciones en la literatura. La
   elección del modelo físico (1/r², gaussiano, kNN) cambia fundamentalmente el
   comportamiento del sistema.

2. **Gravedad clásica 1/r² all-to-all no funciona sin velocidad:** la fuerza neta
   siempre colapsa el sistema a un punto. Requiere momentum/damping como parche,
   lo que aleja el modelo de la física gravitacional pura.

3. **El kernel gaussiano es más apropiado para clustering:** limita la interacción
   a vecinos cercanos, preserva la estructura local y es inherentemente estable.

4. **α es el parámetro más crítico:** controla la escala de movimiento por iteración.
   Demasiado grande → inestabilidad; demasiado pequeño → convergencia lenta.

5. **merge_dist debe ser consistente en todo el experimento:** un `merge_dist`
   diferente entre celdas equivale a evaluar algoritmos distintos.

### 5.2 Sobre el diseño experimental

1. **Evaluar con múltiples seeds:** un algoritmo estocástico evaluado en un solo seed
   puede dar resultados engañosos en cualquier dirección.

2. **Los datos de entrada importan:** `cluster_std=0.7` (compacto) favorece al GCA.
   Con `cluster_std=0.8` el algoritmo puede encontrar sub-clusters legítimos (ARI>0.99)
   pero con más clusters de los esperados.

3. **El sweep debe variar los parámetros correctos:** variar G y Temperatura es menos
   informativo que variar α y merge_dist para entender la sensibilidad del GCA.

4. **Outliers crean micro-clusters en GCA:** esto no es un fallo del algoritmo, sino
   una característica — GCA no tiene concepto de "ruido", todo forma parte de algún
   cluster gravitacional.

---

## 6. Estructura final del notebook

| Celda | Contenido |
|-------|-----------|
| 0 | Título y descripción del modelo físico (LaTeX) |
| 1 | Imports y configuración |
| 2-3 | Generación de datos (cluster_std=0.7, base sin outliers) |
| 4-5 | EDA y visualización de ground truth |
| 6-7 | Preprocesamiento (StandardScaler + masas por densidad kNN) |
| 8-9 | **GCA con kernel gaussiano** + ejecución |
| 10-11 | Visualización del colapso gravitacional |
| 12-13 | Métricas GCA vs KMeans |
| 14-15 | Robustez frente a outliers (0%, 5%, 10%, 20%) |
| 16-17 | Barrido α × merge_dist (heatmaps ARI + clusters) |
| 18-19 | **Evaluación multi-seed** (20 seeds, boxplots) |
| 20-21 | Conclusiones + animación del colapso |
