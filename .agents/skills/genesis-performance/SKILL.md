---
name: genesis-performance
description: Verificar umbrales de    rendimiento de GÉNESIS. Usar cuando se modifiquen funciones en el hot path:from_dense_buf,geometric_product,HNSWsearch, Kuramoto step, o WitnessBuilder.NO usar para cambios que  solo toquen cohomology.rs, lsh.rs, o manifold.rs sin tocar loops de cálculo   numérico.
allow_implicit_invocation: true
---

Ejecuta los benchmarks relevantes al cambio:
cargo bench -p [crate] -- [función_modificada] --output-format bencher

Umbrales duros. Si se superan, el cambio introduce regresión:

-   from_dense_buf: > 200ns → regresión
-   geometric_product sparse: > 500ns → regresión
-   HNSW search 100k: > 10ms → regresión (revisar random_level)
-   Kuramoto step N=10k: > 1ms → regresión
-   Proof generation 4KB: > 100µs → regresión
