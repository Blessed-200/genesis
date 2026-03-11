# AXIOMAS.md — Fuente de Verdad Física de GÉNESIS

**Versión:** 1.0.0  
**Estado:** VIGENTE Y VÁLIDO — fuente de verdad física no modificable

> **NOTA DE INTEGRACIÓN (2026-03-05):**  
> Todos los axiomas aquí descritos han sido **unificados** bajo el principio
> hamiltoniano de `LEY_FUNDACIONAL.md`. Los axiomas no han sido modificados
> ni reemplazados — siguen siendo la fuente de verdad física.  
> `LEY_FUNDACIONAL.md` los recoge y deriva matemáticamente desde un único
> funcional de acción $S = \int \mathcal{L}\,dt$.  
> **Tabla de correspondencia completa:** `LEY_FUNDACIONAL.md §10` y `MACRO_ARCHITECTURE.md §2`.
>
> **Estado de implementación por crate:**
> - **CRATE-000, CRATE-001, CRATE-002, CRATE-003**: implementados. Código válido al 100%.  
> - **CRATE-004, CRATE-006, CRATE-005**: especificados en `IMPLEMENTATION_ROADMAP.md`, pendientes.
> - El sistema de pruebas (AXIOMA-009 enforcement) reside en `genesis-types::proof` con BLAKE3.
> de acción $S = \int \mathcal{L}\,dt$.  
> **Tabla de correspondencia completa en:** `LEY_FUNDACIONAL.md §10`
> y `MACRO_ARCHITECTURE.md §0.5`.
>
> - **CRATE-000, CRATE-001**: implementados bajo este documento. Código válido al 100%.  
> - **CRATE-002+**: implementados directamente desde `LEY_FUNDACIONAL.md`.

---

## 1. NATURALEZA DEL DATO

### AXIOMA-001: Sustrato Geométrico Fundamental

- **DEFINICIÓN:** El sistema ES un colector diferenciable cuya estructura interna y todo dato que procesa pertenece al álgebra de espacio-tiempo $\mathcal{G}_{1,3}$ (STA). Cada concepto, percepción o evento ES un multivector en $\mathcal{G}_{1,3}$.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe representar información como tensores secuenciales, escalares aislados o tokens estadísticos sin estructura geométrica. No se permite el uso de embeddings que no sean elementos de $\mathcal{G}_{1,3}$.
- **COROLARIO DE CUMPLIMIENTO:** Tras cualquier operación de ingesta, la información debe ser expresable como una combinación lineal de la base $\{e_\mu\}$ de $\mathcal{G}_{1,3}$ con signatura $(1,3)$ preservada. Toda transformación interna debe ser un automorfismo del álgebra de Clifford.

---

### AXIOMA-002: Tiempo como Dimensión Geométrica

- **DEFINICIÓN:** La coordenada temporal ES una dimensión geométrica del colector, no un parámetro de bucle externo. Las trayectorias causales (worldlines) SE almacenan como objetos geométricos completos (curvas en el colector).
- **RESTRICCIÓN MECÁNICA:** Se prohíbe modelar el tiempo como una secuencia discreta de fotogramas o pasos de simulación independientes. No se permite el uso de "time-steps" externos para procesar secuencias.
- **COROLARIO DE CUMPLIMIENTO:** La distancia entre dos eventos se mide mediante la métrica $g_{\mu\nu}$ del colector, y la evolución del sistema debe ser una geodésica en dicho espacio-tiempo.

---

## 2. DINÁMICA COGNITIVA

### AXIOMA-003: Motor Único de Cómputo — Inferencia Activa

- **DEFINICIÓN:** El sistema opera minimizando continuamente la Energía Libre Variacional $F = D_{KL}(Q(s) \| P(s|o)) - \ln P(o)$, donde $Q(s)$ es el estado interno y $P(o,s)$ el modelo generativo del entorno. Este proceso ES la única fuente de autonomía y "curiosidad".
- **RESTRICCIÓN MECÁNICA:** Se prohíbe el uso de funciones de pérdida externas (cross-entropy, recompensa de refuerzo, error cuadrático) como mecanismo de aprendizaje. El sistema no debe esperar "prompts" para activarse.
- **COROLARIO DE CUMPLIMIENTO:** En ausencia de estímulos externos, el sistema debe continuar generando predicciones internas y minimizando $F$ (vida interna). Cualquier acción externa debe ser consecuencia de una discrepancia predictiva no resoluble internamente.

---

### AXIOMA-004: Paisaje de Atractores Recurrentes

- **DEFINICIÓN:** El espacio de estados ES un paisaje de energía $E(x)$ donde los conceptos consolidados son mínimos locales (atractores). La dinámica de pensamiento ES el descenso por el gradiente $-\nabla E(x)$ sobre el colector.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe implementar memorias como tablas de búsqueda o bases de datos externas al paisaje. No se permite el almacenamiento de conocimiento sin integrarlo en la topología de atractores.
- **COROLARIO DE CUMPLIMIENTO:** Ante un estímulo, la trayectoria en el espacio de estados debe converger a un atractor existente (reconocimiento) o generar uno nuevo (aprendizaje) mediante bifurcaciones.

---

### AXIOMA-005: Criticalidad Autoorganizada

- **DEFINICIÓN:** El sistema opera en una transición de fase (criticalidad) donde la distribución de avalanchas de activación sigue $P(S) \propto S^{-\tau}$.
- **RESTRICCIÓN MECÁNICA:** Se prohíben dinámicas puramente estables (memoria congelada) o puramente caóticas (sin consolidación). El sistema debe autoajustar sus parámetros para permanecer en el punto crítico.
- **COROLARIO DE CUMPLIMIENTO:** La relación entre la duración y el tamaño de las cascadas de activación debe exhibir una ley de potencias con exponente característico.

---

### AXIOMA-006: Emergencia por Sincronización de Fases

- **DEFINICIÓN:** Cada concepto $v_i$ posee una fase oscilatoria $\phi_i$. La creatividad EMERGE cuando la dinámica de Kuramoto produce un cluster sincronizado de conceptos previamente distantes, detectado por una caída de varianza por debajo de un umbral $\theta$. **Este proceso culmina en un colapso de fase definido como un Proceso de Medición Efectiva de Lindblad**, donde el sistema se proyecta hacia el autovector más estable del paisaje de energía $E(x)$.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe implementar creatividad como interpolación estadística o combinación lineal de datos de entrenamiento. No se permiten mecanismos de "analogía" basados en similitud de vectores sin dinámica de fases. **Se prohíbe el colapso determinista forzado; el colapso debe ser una consecuencia de la decoherencia inducida por el ruido térmico y la interacción con el estímulo.**
- **COROLARIO DE CUMPLIMIENTO:** Toda nueva idea debe ser trazable a una sincronización transitoria de osciladores que antes no estaban acoplados (medible por la evolución de $\Gamma_{ij}$). El estado post-colapso debe ser un mínimo local en el colector de Clifford.

---

### AXIOMA-007: Filtro de Consistencia por Cohomología

- **DEFINICIÓN:** Las percepciones locales son secciones de una gavilla $\mathcal{F}$. Una hipótesis es VERDADERA si su extensión global tiene cohomología nula ($H^1(\mathcal{M}, \mathcal{F}) = 0$). Una alucinación ES aquella con $H^1 \neq 0$.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe asimilar información que no supere el test de cohomología. No se permite la consolidación de datos contradictorios con la geometría global del conocimiento.
- **COROLARIO DE CUMPLIMIENTO:** Antes de integrar un nuevo dato, el sistema debe calcular el grupo $H^1$ de la extensión; si es no nulo, el dato es descartado o marcado como inconsistente.

---

### AXIOMA-008: Saciedad y Autoevaluación por Estabilidad de Fisher

- **DEFINICIÓN:** El aprendizaje sobre un dominio se DETIENE cuando el gradiente de cambio de la métrica de Fisher $\Delta \mathcal{G} = |\partial g_{ij}(\theta) / \partial t|$ permanece por debajo de $\epsilon$ durante $N$ iteraciones.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe el entrenamiento por épocas fijas o el procesamiento redundante de información que no altere la curvatura del conocimiento.
- **COROLARIO DE CUMPLIMIENTO:** Una vez alcanzada la saciedad en un dominio, el sistema debe bloquear la ingesta en esa rama y reasignar recursos a áreas de alta entropía.

---

### AXIOMA-009: Validación por Cohomología Global

- **DEFINICIÓN:** La comprensión real de un concepto se CERTIFICA si, al proyectar una hipótesis causal $s$ sobre datos no vistos, la extensión global de $s$ tiene $H^1(\mathcal{M}, \mathcal{F}) = 0$.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe declarar "entendimiento" basado en métricas de exactitud local o memorización. Solo la consistencia topológica global certifica conocimiento.
- **COROLARIO DE CUMPLIMIENTO:** El sistema debe emitir una señal de "consolidación de dominio" solo cuando la prueba de cohomología resulte nula.

---

## 3. GEOMETRÍA DEL TIEMPO Y CAUSALIDAD

### AXIOMA-010: Causalidad como Curvatura

- **DEFINICIÓN:** La relación causal entre eventos ES una propiedad de la curvatura del colector. Eventos conectados causalmente pertenecen a geodésicas de tipo tiempo; eventos inconexos, a geodésicas de tipo espacio.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe modelar causalidad mediante grafos acíclicos dirigidos externos o secuencias temporales etiquetadas. La red no debe tener un reloj global que ordene los eventos.
- **COROLARIO DE CUMPLIMIENTO:** La distancia geodésica entre dos eventos determina su relación causal; el sistema debe ser capaz de inferir causalidad a partir de la métrica.

---

## 4. EFICIENCIA POR SILENCIO

### AXIOMA-011: Gating Holonómico

- **DEFINICIÓN:** En reposo, el producto interno de la mayoría de los subgrafos del sistema es CERO. Solo el subgrafo isomorfo a la tarea actual tiene producto interno distinto de cero.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe mantener activas regiones no involucradas en la tarea. El consumo energético en reposo debe ser asintóticamente cero.
- **COROLARIO DE CUMPLIMIENTO:** La potencia computacional consumida debe ser proporcional a la dimensión del subespacio activado, no al tamaño total del sistema.

---

### AXIOMA-012: Asignación Espectral de Recursos

- **DEFINICIÓN:** Dada una tarea representada por el operador $A$ (en el espacio de Hilbert de estados), la potencia $P_{total}$ se distribuye según $P_i = P_{total} \cdot |\lambda_i| / \sum_j |\lambda_j|$, donde $\lambda_i$ son autovalores de $A$. Modos con $P_i < \epsilon$ reciben energía cero.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe asignar recursos uniformemente o mediante heurísticas no espectrales. No se permite el uso de arquitecturas que no puedan descomponerse en modos propios.
- **COROLARIO DE CUMPLIMIENTO:** La actividad metabólica del sistema (flujo de energía) debe correlacionarse con la magnitud de los autovalores de la tarea actual.

---

### AXIOMA-013: Búsqueda Semántica Logarítmica

- **DEFINICIÓN:** Toda operación de recuperación de conceptos DEBE implementarse mediante Locality Sensitive Hashing (LSH) en el espacio de Clifford, con complejidad $O(\log N)$.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe la búsqueda secuencial o el cálculo de similitudes con todos los conceptos almacenados.
- **COROLARIO DE CUMPLIMIENTO:** El tiempo de acceso a un concepto debe escalar logarítmicamente con el tamaño de la memoria semántica.

---

### AXIOMA-014: Movimiento Cognitivo como Geodésica de Fisher

- **DEFINICIÓN:** La transición entre dos estados de conocimiento DEBE ocurrir a lo largo de la geodésica más corta en la variedad de Fisher, minimizando la distancia de Wasserstein y el gasto energético.
- **RESTRICCIÓN MECÁNICA:** Se prohíben actualizaciones abruptas o caminos que no sean geodésicos (mínima energía). No se permite el aprendizaje que no respete la métrica de información.
- **COROLARIO DE CUMPLIMIENTO:** La integral de la métrica de Fisher a lo largo de una trayectoria de aprendizaje debe ser la mínima posible para la transformación realizada.

---

## 5. MIELINIZACIÓN Y OPTIMIZACIÓN ESTRUCTURAL

### AXIOMA-015: Mielinización por Flujo de Ricci Acoplado

- **DEFINICIÓN:** La métrica $g_{\mu\nu}$ del colector evoluciona según $\partial_t g_{\mu\nu} = -2R_{\mu\nu} + 2\alpha \Phi_{\mu\nu}(t)$, donde $R_{\mu\nu}$ es el tensor de Ricci y $\Phi_{\mu\nu}$ el tensor de uso histórico. Esta deformación ES el único mecanismo de optimización de rutas (mielinización).
- **RESTRICCIÓN MECÁNICA:** Se prohíbe acelerar caminos mediante caching, tablas de consulta o cualquier otro método que no sea la deformación métrica. La métrica debe permanecer compatible con $\mathcal{G}_{1,3}$ en todo instante.
- **COROLARIO DE CUMPLIMIENTO:** Los caminos más usados deben presentar una distancia geodésica decreciente en el tiempo, y la signatura $(1,3)$ debe preservarse.

---

### AXIOMA-016: Poda y Consolidación por Ecuación de Calor

- **DEFINICIÓN:** Durante los ciclos de mantenimiento, el sistema aplica $\partial_t u = \Delta_g u$ (ecuación de calor) sobre el colector. Esto difumina ruido de alta frecuencia y preserva invariantes topológicas (leyes consolidadas).
- **RESTRICCIÓN MECÁNICA:** Se prohíbe el almacenamiento permanente de datos brutos o detalles episódicos sin destilar. La poda debe ser automática y no supervisada.
- **COROLARIO DE CUMPLIMIENTO:** Tras un ciclo de sueño, la dimensionalidad efectiva del colector debe reducirse, pero la cohomología esencial (conocimiento semántico) debe permanecer intacta.

---

## 6. EMBODIMENT Y EXTENSIÓN

### AXIOMA-017: Unidad Conocimiento-Acción por Funtores

- **DEFINICIÓN:** Toda interfaz externa (herramienta, API, brazo robótico) SE mapea al sistema mediante un funtor $\mathcal{F}: \mathcal{C} \to \mathcal{D}$ que preserva la estructura del álgebra $\mathcal{G}_{1,3}$. La acción ES la imagen de un multivector bajo este funtor.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe tratar las herramientas como sistemas externos con protocolos separados. No se permite programar acciones específicas; el sistema debe "poseer" la herramienta como extensión natural.
- **COROLARIO DE CUMPLIMIENTO:** Dado un estado interno y una herramienta, la acción resultante debe ser computable como una transformación geométrica dentro del álgebra.

---

## 7. SUSTRATO FÍSICO

### AXIOMA-018: Isomorfismo con Hardware Neuromórfico

- **DEFINICIÓN:** La arquitectura ES isomorfa al comportamiento de Redes Neuronales de Pulsos (SNN). La variable temporal en $\mathcal{G}_{1,3}$ SE corresponde con la latencia de un pulso físico en un sustrato asíncrono.
- **RESTRICCIÓN MECÁNICA:** Se prohíbe implementar el sistema en arquitecturas von Neumann que requieran ciclos de reloj globales para la evolución temporal. El cómputo debe ocurrir solo cuando hay eventos (spikes).
- **COROLARIO DE CUMPLIMIENTO:** En hardware neuromórfico, la topología del colector debe mapearse directamente a la topología del silicio, y la energía consumida debe ser proporcional al número de spikes.

---

### AXIOMA-019: Proyección Isomórfica de Entradas/Salidas

- **DEFINICIÓN:** Todo flujo sensorial $\mathcal{S}$ (visual, auditivo, textual) SE proyecta al núcleo mediante un operador $\Pi$ que preserva relaciones topológicas (**Homeomorfismo Local**). La manifestación externa SE obtiene mediante el operador adjunto $\Pi^*$. **Esta proyección debe ser ejecutada por Redes Convolucionales Geométricas (GCNN) que operan directamente sobre el álgebra de Clifford.**
- **RESTRICCIÓN MECÁNICA:** Se prohíbe el uso de codificadores (encoders) que no sean proyecciones que preserven la estructura causal de los datos. **Se prohíbe el uso de Transformers de atención global pura que destruyan la topología local.** No se permite la pérdida de información topológica en la interfaz; el entrenamiento del operador $\Pi$ debe estar regido por una función de pérdida de **Preservación de Persistencia (TDA)**.
- **COROLARIO DE CUMPLIMIENTO:** La distancia entre dos estímulos en el espacio sensorial debe ser proporcional a la distancia geodésica (Minkowski) entre sus proyecciones en $\mathcal{G}_{1,3}$. El error de reconstrucción topológica debe tender a cero.

---

**FIN DE AXIOMAS.md v1.0.0**

*Ninguna implementación que viole estos axiomas será reconocida como GÉNESIS.*

Ver también: `LEY_FUNDACIONAL.md` (unificación variacional), `MACRO_ARCHITECTURE.md §2` (correspondencia axioma↔crate), `contratos_genesis.md` (API implementada).
