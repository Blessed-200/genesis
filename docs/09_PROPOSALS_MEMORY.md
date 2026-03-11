# PROPOSALS_MEMORY.md — Acumulador de Propuestas para Crates Futuros
**Estado:** DOCUMENTO VIVO  
**Última actualización:** v5.5.0  
**Propósito:** Registro permanente de propuestas arquitectónicas aprobadas para fases futuras.
Cada Ingeniero Principal que trabaje en este proyecto debe leer este documento primero.

---

## PROPUESTAS APLICADAS (CRATE-000 → CRATE-003)

### P-001 — Adaptive synchrony threshold [APLICADO v5.4.0]
`synchrony_order_fast`: serial para N < 4096, parallel para N ≥ 4096.
Elimina overhead de rayon para redes pequeñas (83% reducción latencia en CI).

### P-002 — Const fn idiomatic [APLICADO v5.5.0]
14 funciones promovidas a `const fn` donde MSRV 1.75 lo permite.
`is_fresh`, `try_new`, `cardinality`, `is_internal_drive`, etc.

### P-003 — Sorted Vec en edge_map H¹ [APLICADO v5.3.0]
`IncrementalH1State::edge_map`: `HashMap<u128, u32>` → `Vec<(u128, u32)>` con binary search.
O(log E) lookup, cache-friendly, AGENTS compliance.

### P-004 — Wolfram-derived Kuramoto constants [APLICADO v5.4.0]
`KURAMOTO_CRITICAL_COUPLING = 0.01` = 2γ/λ_max derivado analíticamente.
`KURAMOTO_GAMMA_WIDTH = 0.1`, `KURAMOTO_LAMBDA_MAX_EXPECTED = 20.0`.

### P-005 — CRATE-004 prerequisite APIs [APLICADO v5.4.0]
`remove_oscillator`, `amplitude_norm`, `set_amplitude_all_grades` (Kuramoto).
`beliefs_raw`, `remove_node` (VFEMinimizer).
`remove_node` (HnswGraph + ManifoldCollector).
`GenesisError::NodeNotFound`.

---

## PROPUESTAS PARA CRATE-004 — genesis-evolution

### P-010 — Inelastic Concept Fusion (WormholeCollapse mejorado)
**Fuente:** Blueprint CRATE-004, propuesta de ingeniería anterior.
**Descripción:** `inelastic_concept_fusion(survivor, absorbed)` en `wormhole.rs`.
- Fase Kuramoto fusionada: φ_merged = weighted_mean(φ_survivor, φ_absorbed)
- VFE belief fusionada: Bayesian merge de distribuciones Gaussianas
- HNSW: redirect edges de absorbed → survivor, entonces `remove_node(absorbed)`
- Señal: `CollapseReason::Redundancy` activa cuando H_compresión > θ_r
**APIs prerequisito ya implementadas:** ✅ todas en v5.4.0
**AX-ID:** LEY_FUNDACIONAL §3.7, §6.5

### P-011 — Coherence-weighted Ricci Flow
**Fuente:** Análisis de código Python presentado + literatura Ginzburg-Landau.
**Descripción:** Añadir término de coherencia de fases al flujo de Ricci:
```
dg/dt = -2·K_ij·g + 2α·Φ_ij + β·coherence_signal(i,j)
coherence_signal(i,j) = exp(-var(φ_neighbours) × λ)
```
Cuando la varianza de fases en el vecindario es BAJA, fortalecer la conexión.
Esto acopla genesis-dynamics (fases Kuramoto) con genesis-evolution (métrica).
**Implementar en:** `ricci.rs::DiscreteRicciFlow::step()`
**AX-ID:** AXIOMA-015, H_estructura

### P-012 — Gram-Schmidt con condición KKT (Wolfram PASO 4)
**Fuente:** WOLFRAM_PHYSICS_IMPLEMENTATION.md §4 + LEY_FUNDACIONAL §3.8, §5.7, §6.6.
**Descripción:** `should_expand_dimensionality(obs, proj, fisher_precision, n_nodes) -> bool`
```rust
residual_energy = Σᵢ fisher_precision[i] * (obs[i] - proj[i])²
c_dim = LAMBDA_DIM_FIXED + LAMBDA_DIM_LOG * ln(n_nodes)
return residual_energy > c_dim
```
Sustituye `JL_RESIDUAL_EXPANSION_DELTA` como criterio principal de expansión.
**Implementar en:** `gram_schmidt.rs::GramSchmidtExpander::try_expand()`
**AX-ID:** LEY_FUNDACIONAL §3.8, §5.7, DimensionalAdmission

### P-013 — Rotor dominante + proyección Lindblad (Wolfram PASO 3)
**Fuente:** WOLFRAM_PHYSICS_IMPLEMENTATION.md §3.
**Descripción:** Mapeo 5-grados Kuramoto → 16-dim SparseCliffordVector:
```rust
fn extract_dominant_rotor(neighbours: &[[f64;4]]) -> [f64;6]  // ángulos θ_ij
fn project_sync_to_clifford(amplitudes, phases, theta_ij) -> SparseCliffordVector
```
`fast_4x4_top_two_eigenvectors` via iteración de potencia 4×4 (NO placeholder).
**Implementar en:** `wormhole.rs`
**AX-ID:** AXIOMA-006, H_información

### P-014 — H_compresión señal WormholeCollapse
**Fuente:** LEY_FUNDACIONAL §3.7, §6.5.
**Descripción:** `compute_h_compression(phases, hnsw, ε_r, θ_r) → (f64, Vec<(NodeId, NodeId)>)`.
Usa fases Kuramoto ya calculadas (zero extra cost).
Cuando redundancy > KAPPA_REDUNDANCY → emitir señal `CollapseReason::Redundancy`.
**Implementar en:** `compression.rs` (módulo nuevo)
**AX-ID:** LEY_FUNDACIONAL §3.7, AXIOMA-016

---

## PROPUESTAS PARA CRATE-005 — genesis-io

### P-020 — Topological Persistence Loss (Wolfram PASO 5)
**Fuente:** WOLFRAM_PHYSICS_IMPLEMENTATION.md §5.
**Descripción:** `apply_topological_gradient(points, wasserstein_gradients, lr)`.
Preserva homología H⁰ y H¹ durante la proyección sensorial.
**Implementar en:** `drivers/genesis-io/src/projection.rs`
**AX-ID:** AXIOMA-019

---

## PROPUESTAS PARA CRATE-006 — genesis-consciousness

### P-030 — H_dualidad + Ω_dual
**Fuente:** LEY_FUNDACIONAL §3.6, §4.
**Descripción:**
```rust
fn compute_h_duality(metric: &EdgeMetric, fisher: &FisherEdgeMetric) -> f64
fn compute_omega_dual(h_duality, h_duality_max) -> f64
```
En `duality.rs`. Ω_dual = 1 - H_dualidad/H_max_dualidad ∈ [0,1].
**AX-ID:** LEY_FUNDACIONAL §3.6, §5.6, DualityConsistency

### P-031 — GlobalObserver::compute_omega_v11
**Fuente:** LEY_FUNDACIONAL §4, IMPLEMENTATION_ROADMAP FASE 6.
Ω = 0.25·r_sync + 0.20·|K_avg| + 0.20·(1/ΔG+ε) + 0.15·H¹=0 + 0.10·λ₂ + 0.10·Ω_dual.
**AX-ID:** LEY_FUNDACIONAL §4

### P-032 — DecisionSignal::ForceDualUpdate
**Fuente:** IMPLEMENTATION_ROADMAP FASE 6.
Nueva variante: `ForceDualUpdate { nodes: Vec<NodeId> }`.
Se emite cuando Ω_dual < DUAL_COHERENCE_THRESHOLD.

---

## PROPUESTAS ARQUITECTÓNICAS DE ESCALA (POST CRATE-006)

### P-040 — Sparse Active Graph / Event-Driven Simulation
**Fuente:** Propuesta_1.md + CLOUD_PLATFORM_ARCHITECTURE §1.
**Descripción:** Separar ColdGraph (pasivo, 10⁹ nodos) de ActiveGraph (dinámico, 10⁶ nodos).
Event-driven: solo nodos con energía > ACTIVATION_THRESHOLD ejecutan Kuramoto + VFE.
**Implementar en:** scheduler del genesis-compute-engine (cloud layer).
**AX-ID:** AXIOMA-011 (gating holonómico)

### P-041 — Predictive Concept Graph (genesis-predictive crate)
**Fuente:** Propuesta_2.md.
**Descripción:** Crate adicional que OBSERVA el grafo y propone nodos anticipados.
- `detect_pattern(neighbourhood) → Option<Pattern>`
- `predict_concept(pattern) → ConceptNode`
- `materialize_if_needed(node) → bool`
No modifica interfaces actuales. Depende de genesis-types + genesis-topology + genesis-dynamics.
**Estado:** Después de CRATE-006.

### P-042 — Cohomología predictiva (H¹ como motor de exploración)
**Fuente:** Propuesta alternativa del proyecto.
`dim(H¹) > 0` → zona de incertidumbre → exploración cognitiva dirigida.
Integrar con H_teleología: penalizar lagunas topológicas, recompensar exploración de H¹>0.
**AX-ID:** AXIOMA-007

### P-043 — Probabilistic H¹ validator O(E log N) (Wolfram PASO 2)
**Fuente:** WOLFRAM_PHYSICS_IMPLEMENTATION.md §2.
Lazy Random Walk sobre aristas (1-cochains) vía triángulos compartidos.
Si gap espectral > ε → H¹=0 con probabilidad alta. Reemplaza eliminación gaussiana O(E³).
**Implementar en:** `incremental_cohomology.rs` como `h1_is_zero_fast_probabilistic()`.
**AX-ID:** AXIOMA-007, AXIOMA-009

### P-044 — CompressedNode + Elias-Fano topology (escala > 10⁷ nodos)
**Fuente:** Sugerencias de arquitectura de data, análisis de auditores.
```rust
#[repr(transparent)]
pub struct CompressedNode(u64);  // 8 bytes: geom_type(8bits) | arena_index(56bits)
pub struct GeometricArenas { points, planes, spheres, motors }
pub struct BVCompressedEdges { data: Vec<u64>, offsets: Vec<u64> }  // ~3 bits/edge
```
De 160 bytes/nodo → 8 bytes/nodo. De 200 GB topología → 15 GB.
**Estado:** Post-CRATE-006, CLOUD arquitectura. No impacta crates cognitivos.

---

## CRITERIO GoNoGo POR FASE

### CRATE-004 GoNoGo:
✅ `inelastic_concept_fusion` compila con APIs de FIX-H
✅ `WormholeCollapse` acepta `CollapseReason::Redundancy`
✅ `GramSchmidtExpander` verifica `DimensionalAdmission` (condición KKT)
✅ `DiscreteRicciFlow` con Ollivier-Ricci + coherence_signal
✅ `FisherEdgeMetric` expuesta desde genesis-evolution (para genesis-consciousness)
✅ Todos los tests de CRATE-004 pasan
✅ `cargo doc --workspace` 0 warnings
✅ `cargo build --workspace` 0 warnings

### CRATE-005 GoNoGo (genesis-io):
✅ GCNN projection O preserving H⁰/H¹
✅ `apply_topological_gradient` implementado
✅ Functors AXIOMA-017
✅ Full integration test pipeline cognitivo

### CRATE-006 GoNoGo (genesis-consciousness):
✅ `compute_omega_v11` con Ω_dual
✅ `ForceDualUpdate` emitido cuando Ω_dual < threshold
✅ `H_dualidad` y `H_compresión` integrados en Ω
✅ Sistema autónomo: minimiza H_total sin input externo
✅ AXIOMA-005 (SOC): el sistema no puede congelarse

---

## NOTAS DE IMPLEMENTACIÓN CRÍTICAS

### Sobre `fast_4x4_top_two_eigenvectors`:
El placeholder `([1,0,0,0], [0,1,0,0])` de Wolfram PASO 3 NO es suficiente.
Implementar power iteration con deflation:
```rust
fn power_iteration(cov: &[[f64;4];4], max_iter: u32) -> [f64;4] {
    let mut v = [1.0/2.0_f64.sqrt(); 4];
    for _ in 0..max_iter {
        let mut w = [0.0f64; 4];
        for i in 0..4 { for j in 0..4 { w[i] += cov[i][j] * v[j]; } }
        let norm = w.iter().map(|x| x*x).sum::<f64>().sqrt();
        if norm < 1e-12 { break; }
        for i in 0..4 { v[i] = w[i] / norm; }
    }
    v
}
```

### Sobre SparseCliffordVector en CRATE-004:
Acceder a campos via `osc.phases[g]` y `osc.amplitudes[g]` directamente.
Los índices de blade para vectores grado-1: `GRADE1_BLADE_INDICES = [1, 2, 4, 8]`.
Usar `SparseCliffordVector::from_dense(&coeffs)` para construir desde arrays.

### Sobre dependencias circulares:
`compression.rs` en genesis-evolution puede acceder a genesis-topology (HNSW) → OK.
`duality.rs` en genesis-consciousness accede a genesis-evolution + genesis-dynamics → OK.
NUNCA: genesis-topology → genesis-evolution. NUNCA: genesis-dynamics → genesis-evolution.

### Sobre el benchmark guardrail:
`BASELINE_SYNCHRONY_ORDER_FAST_1000_NS = 20_000.0` (serial path para N=1000).
Para N ≥ 4096 (rayon): el guardrail debería ser diferente.
Crear benchmark separado para el path paralelo: `synchrony_order_fast_10000_nodes`.

---

**FIN DE PROPOSALS_MEMORY.md**
Actualizado: v5.5.0 | Principal Engineer
