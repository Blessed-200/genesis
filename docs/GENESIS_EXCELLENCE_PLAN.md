# PLAN DE EXCELENCIA — GÉNESIS CRATE-000 → CRATE-003
## Objetivo: Nivel DeepMind / SOTA. Sin huecos. Sin poesía.

**Estado actual medido (v5.4.0):**
- Tests: 411 pasados, 0 fallidos
- Warnings build: 0
- todo!/unimplemented!: 0 en prod
- Unsafe sin SAFETY: 12 bloques
- Fuzz targets: 0 (no existe /fuzz/)
- Property tests: 1 (solo genesis-math product.rs)
- Documentación API pública: ~0% tiene doc completo con invariante matemático
- cargo fmt: no disponible en toolchain local (verificar en CI)
- Benchmarks guardrails: 1 aún frágil (synchrony_order_fast threshold)

---

## PROTOCOLO DE EJECUCIÓN

**Codex ejecuta las tareas en orden. Cada tarea produce un bloque de output.
Ese output se pega en genesis_codex_log.txt que me traes a mí.**

El formato de cada bloque en el log:
```
=== TASK-XX: [nombre] ===
COMANDO: <exactamente lo que se ejecutó>
STATUS: PASS | FAIL | PARTIAL
OUTPUT:
<stdout/stderr relevante>
=== END TASK-XX ===
```

---

## FASE 1 — STATIC ANALYSIS TOTAL (estilo lab)

### TASK-01: rustfmt exhaustivo
```bash
cargo fmt --version || rustup component add rustfmt
cargo fmt --all
git diff --stat
```
Registrar: cuántos archivos cambiaron, qué archivos.

### TASK-02: clippy PEDANTIC completo
El objetivo: zero warnings a nivel PEDANTIC. Esto es lo que diferencia
hobby de ingeniería de producción.
```bash
cargo clippy --workspace --all-targets --all-features \
  -- -D warnings \
  -W clippy::pedantic \
  -W clippy::nursery \
  -A clippy::module_name_repetitions \
  -A clippy::must_use_candidate \
  -A clippy::missing_errors_doc \
  2>&1 | tee /tmp/clippy_pedantic.txt

grep "^error" /tmp/clippy_pedantic.txt | sort | uniq -c | sort -rn | head -30
```
Registrar: lista completa de errores únicos con count.

### TASK-03: rustdoc sin warnings
```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps 2>&1 | \
  grep "warning:\|error:" | sort | uniq | head -20
```

### TASK-04: cargo deny (licencias + duplicados)
```bash
cargo install cargo-deny --locked 2>/dev/null || true
cat deny.toml 2>/dev/null || echo "NO deny.toml"
cargo deny check 2>&1 | tail -20
```

### TASK-05: unsafe surface audit manual
```bash
grep -rn "unsafe " --include="*.rs" core/ shared/ | \
  grep -v "// SAFETY:" | grep -v "#\[test\]" | \
  grep -v "doc\|\.md" | grep "unsafe {"
```
Registrar: lista exacta de unsafe sin SAFETY comment.

---

## FASE 2 — CORRECTITUD ALGEBRAICA PROFUNDA

### TASK-06: Verificar leyes del álgebra de Clifford G(1,3)
Esto es lo que NO tiene GÉNESIS y que DeepMind exigiría.
```bash
cat >> /tmp/test_algebra.rs << 'RUST'
// Leyes que DEBEN cumplirse en G(1,3):
// 1. Associatividad: (AB)C = A(BC)
// 2. Distributividad: A(B+C) = AB + AC
// 3. Contracción: e_i * e_i = g(e_i, e_i) (signatura Minkowski)
// 4. Anti-conmutación: e_i * e_j = -e_j * e_i (i≠j)
// 5. Involución de reversa: rev(AB) = rev(B) * rev(A)
// 6. Norma: ||A||² ≥ 0 para todo A de grado par
// 7. Idempotencia de proyección de grado
RUST
```
Ejecutar en genesis-math:
```bash
cd core/genesis-math
cargo test -- --test-output immediate 2>&1 | grep -E "FAILED|ok|IGNORED" | tail -20
```
Registrar: qué tests cubren qué ley algebraica. Qué leyes NO están cubiertas.

### TASK-07: Property tests — Álgebra G(1,3) con proptest
Verificar si proptest está en Cargo.toml y qué cubre:
```bash
grep -rn "proptest\|prop_assert\|arbitrary" core/genesis-math/Cargo.toml \
  core/genesis-math/src/ --include="*.toml" --include="*.rs" | head -10
cargo test -p genesis-math --features properties -- prop 2>&1 | tail -10
```

### TASK-08: Invariantes físicos — test suite completo
```bash
cargo test --workspace -- \
  minkowski \
  signature \
  cohomology \
  lambda2 \
  kuramoto \
  synchrony \
  proof \
  2>&1 | grep -E "test.*\.\.\. (ok|FAILED|ignored)" | head -40
```

### TASK-09: Test de regresión numérica — producto geométrico
Valores conocidos de G(1,3) que DEBEN ser exactos:
```bash
# e1*e1 = +1 (timelike), e2*e2 = -1 (spacelike)
# e1*e2 = e12 = -e2*e1
# (e1*e2)*(e2*e1) = -1 (inverso en bivector)
cargo test -p genesis-math -- \
  basis_squares \
  grade_involution \
  reverse_product \
  metric_distance \
  2>&1 | grep -E "ok|FAILED"
```

---

## FASE 3 — PROPERTY TESTING EXHAUSTIVO (proptest / quickcheck)

### TASK-10: Instalar proptest y crear suite para genesis-types
Si no está, añadir a dev-dependencies y crear tests:
```bash
grep "proptest" shared/genesis-types/Cargo.toml
cargo test -p genesis-types -- node_id proof fisher 2>&1 | tail -20
```

### TASK-11: Property tests para genesis-topology
```bash
# ¿Existe property testing para HNSW? ¿Para H¹?
grep -rn "proptest\|quickcheck" core/genesis-topology/ --include="*.rs" | wc -l
cargo test -p genesis-topology -- hnsw cohomology lambda 2>&1 | tail -20
```

### TASK-12: Smoke test de las nuevas APIs (FIX-H)
Las APIs que añadimos para CRATE-004 DEBEN tener tests propios:
```bash
cargo test -p genesis-dynamics -- remove_oscillator amplitude_norm beliefs_raw 2>&1
cargo test -p genesis-topology -- remove_node 2>&1
```
Registrar: PASS / FAIL. Estas APIs aún no tienen tests.

---

## FASE 4 — FUZZ TESTING (Plan A: local / Plan B: CI)

### TASK-13: Setup de fuzz targets
```bash
cargo install cargo-fuzz --locked 2>/dev/null || true
cargo fuzz --version 2>/dev/null || echo "FUZZ_NOT_AVAILABLE"
```
Si disponible:
```bash
# Target 1: geometric product con inputs arbitrarios
# Target 2: HNSW insert/search con vectores aleatorios
# Target 3: Proof generation/verification
# Target 4: VFEMinimizer con NodeIds extremos
ls fuzz/ 2>/dev/null || echo "fuzz/ directory does not exist"
```
Registrar si cargo-fuzz funciona en el entorno de Codex.

### TASK-14: Corpus de edge cases (sin fuzz, con hardcoded cases)
```bash
# Casos extremos que deben no crashear:
# - NodeId::MAX
# - SparseCliffordVector con todos los coeficientes = 0
# - SparseCliffordVector con NaN/Inf (debe rechazar)
# - HNSW search en grafo vacío
# - VFE con belief precision=0
cargo test -p genesis-types -- edge_cases sentinel max 2>&1 | tail -10
cargo test -p genesis-math -- zero_vector nan_input 2>&1 | tail -10
cargo test -p genesis-topology -- empty_graph sentinel 2>&1 | tail -10
cargo test -p genesis-dynamics -- empty_network extreme 2>&1 | tail -10
```

---

## FASE 5 — BENCHMARKS HONESTOS Y DOCUMENTADOS

### TASK-15: Ejecutar todos los benchmarks con baseline honesta
```bash
cargo bench -p genesis-types --no-run 2>&1 | tail -5
cargo bench -p genesis-math --bench geometry -- --test 2>&1 | tail -10
```
NO ejecutar el bench de dynamics completo (guardrail puede fallar en CI runner).
Solo compilar para verificar que compilan:
```bash
cargo bench --workspace --no-run 2>&1 | grep "error\|Compiling" | tail -10
```

### TASK-16: Verificar que guardrail de synchrony es realista
```bash
grep -n "BASELINE_SYNCHRONY_ORDER_FAST\|35_000\|20_000" \
  core/genesis-dynamics/benches/dynamics.rs
```
El threshold 20_000ns — ¿es correcto para el hardware de Codex?
```bash
# Medir en el entorno real de Codex:
timeout 30 cargo bench -p genesis-dynamics --bench dynamics \
  -- synchrony_order_fast 2>&1 | grep "time:\|ns/iter\|guardrail" | head -5
```

---

## FASE 6 — DOCUMENTACIÓN NIVEL LAB

### TASK-17: Auditar APIs públicas sin doc matemático
```bash
# Función pública sin /// doc comment = BLOQUEANTE en nivel DeepMind
grep -rn "^\s*pub fn" core/genesis-math/src/product.rs | head -10
grep -rn "^\s*pub fn" core/genesis-topology/src/manifold.rs | head -10
grep -rn "^\s*pub fn" core/genesis-dynamics/src/synchrony.rs | head -10
```
Registrar: lista de `pub fn` sin doc comment encima.

### TASK-18: Verificar que contratos genesis_contracts.md están al día
```bash
cat contratos_genesis.md | head -50
# Comparar con APIs actuales
grep -rn "pub fn.*NodeId\|pub fn.*SparseClifford\|pub fn.*Proof" \
  shared/genesis-types/src/ --include="*.rs" | grep -v "test\|doc" | head -20
```

### TASK-19: Missing deny attributes en lib.rs
DeepMind-level requiere en cada crate lib.rs:
```bash
for f in shared/genesis-types/src/lib.rs core/genesis-math/src/lib.rs \
          core/genesis-topology/src/lib.rs core/genesis-dynamics/src/lib.rs; do
  echo "=== $f ==="
  grep "#!\[deny\|#!\[warn\|#!\[forbid" $f
done
```
Debe tener: `#![deny(unsafe_op_in_unsafe_fn)]`, `#![warn(missing_docs)]`
(o justificación de por qué no).

---

## FASE 7 — CONTRATOS DE COMPLEJIDAD (SOTA comparison)

### TASK-20: Verificar complejidades documentadas vs implementadas
```bash
# HNSW: insert debe ser O(log N) amortizado
# lambda2: debe ser O(N * iter) con iter << N
# H¹: incremental must be near-O(α(N)) amortized
grep -rn "O(log\|O(N\|O(K\|O(α\|complexity\|Complexity\|AX-ID" \
  core/genesis-topology/src/hnsw.rs | head -10
grep -rn "O(log\|O(N\|O(K\|complexity" \
  core/genesis-topology/src/manifold.rs | head -10
```

### TASK-21: Medir que HNSW search es realmente O(log N)
```bash
# Insertar N=100, N=1000, N=10000 y medir tiempo de búsqueda
# Si t(10000)/t(100) ≈ log(10000)/log(100) = 2 → O(log N) confirmed
# Si t(10000)/t(100) ≈ 100 → O(N) → FALLO ARQUITECTURAL
cargo test -p genesis-topology -- hnsw_search_scaling 2>&1 | tail -5
```
Si ese test no existe, registrar: TEST_MISSING.

---

## FASE 8 — SAFETY (MIRI / SANITIZERS via CI)

### TASK-22: ¿Está MIRI disponible localmente?
```bash
cargo miri --version 2>&1 || echo "MIRI_NOT_AVAILABLE"
rustup component list --installed | grep miri
```

### TASK-23: Address Sanitizer (si disponible en entorno)
```bash
# Solo genesis-math que tiene unsafe NEON/x86
RUSTFLAGS="-Z sanitizer=address" \
  cargo +nightly test -p genesis-math 2>&1 | tail -10 || echo "NIGHTLY_NOT_AVAILABLE"
```

---

## REGISTRO FINAL QUE DEBE CONTENER genesis_codex_log.txt

Al final de cada fase, Codex añade:
```
=== SUMMARY FASE X ===
PASS: [lista de tasks que pasaron]
FAIL: [lista con motivo exacto]
MISSING: [lo que no se pudo ejecutar y por qué]
ACTION_NEEDED: [qué tiene que corregir la Principal Engineer]
```

---

## PLAN B — GITHUB ACTIONS PARA LO QUE CODEX NO PUEDE

Crear estos workflows adicionales en .github/workflows/:

### genesis_fuzz.yml (ejecutar manualmente o weekly)
```yaml
# Fuzzing con cargo-fuzz — 300 segundos por target
# Targets: geometric_product, hnsw_insert, proof_roundtrip, vfe_compute
# Artifact: corpus + crash reports
```

### genesis_miri.yml (ejecutar en nightly, weekly)
```yaml
# MIRI sobre genesis-math y genesis-types
# usa dtolnay/rust-toolchain@nightly + miri component
# Solo los crates sin dependencias de SIMD condicional
```

### genesis_asan.yml (address sanitizer, weekly)
```yaml
# RUSTFLAGS="-Z sanitizer=address" sobre genesis-math
# usa nightly toolchain
# detecta UAF, buffer overflow en unsafe NEON
```

### genesis_proptest_extended.yml (weekly, 10000 casos por property)
```yaml
# PROPTEST_CASES=10000 cargo test --features properties
# Más casos que en el CI diario para cubrir corner cases raros
```

---

## CRITERIO DE APROBACIÓN (GoNoGo para pasar a CRATE-004)

Para que el Ingeniero Principal declare LISTO:

✅ cargo check --workspace: 0 warnings
✅ clippy --D warnings: 0 errors (pedantic es bonus, no bloqueante)
✅ cargo doc -D warnings: 0 warnings
✅ Todas las leyes algebraicas de G(1,3) tienen test explícito
✅ Todas las APIs de FIX-H tienen al menos 1 test
✅ Unsafe blocks: todos tienen SAFETY comment verificado
✅ HNSW O(log N) confirmado con test de scaling
✅ Benchmarks compilan sin error
✅ contratos_genesis.md refleja APIs actuales

❌ Bloqueantes que impiden GoNoGo:
- Cualquier FAILED en cargo test
- Unsafe sin SAFETY
- API pública sin doc (en crates CRATE-000 y CRATE-001 al menos)
- H¹ test que se cuelga (compute_h1_returns_zero_or_one)
