---
name: genesis-invariants
description: Verificar invariantes físicos del sistema GÉNESIS antes y
  después de cualquier cambio en genesis-math, genesis-topology, o
  genesis-dynamics. Usar cuando se modifiquen tipos algebraicos, la métrica
  HNSW, el flujo de Ricci, o el sistema de Proof.
  NO usar para cambios que solo toquen tests, benchmarks, comentarios,
  o Cargo.toml sin modificar lógica de tipos.
allow_implicit_invocation: true
---

Antes de cualquier cambio, ejecuta:
cargo test --release -p [crate] -- invariant --nocapture

Verifica específicamente:
1. Minkowski signature: coefficients[0] positivo, [1][2][3] negativos
2. HNSW levels: histograma de 1000 inserciones con P(0) ∈ [0.50, 0.65]
3. Proof hash: blake3 de witness == proof.hash
4. lambda2: ≥ 0.1 en grafo conectado de 10 nodos

Si alguno falla después del cambio, el cambio está incompleto.
