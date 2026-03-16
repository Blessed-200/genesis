# OPERATION CRYSTAL FORENSICS — Geometric Product Forensic Performance Report

## Phase 1 — CPU architecture identification

- **CPU model:** Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
- **Logical CPUs:** 3 (cores/socket 3, sockets 1, threads/core 1)
- **SIMD flags detected in `lscpu`:** AVX (`avx`), AVX2 (`avx2`), AVX-512 (`avx512f avx512dq avx512bw avx512vl avx512cd`), FMA (`fma`).
- **Cache sizes (`lscpu`):** L1d 96 KiB (2 instances), L1i 64 KiB (2 instances), L2 2.5 MiB (2 instances), L3 48 MiB (1 instance).
- **Pipeline width:** not reported by `lscpu` in this environment (no direct measured source in repo tooling).
- **FMA unit count:** not reported by `lscpu` in this environment (no direct measured source in repo tooling).

## Phase 2 — Benchmark execution

- Command requested by mission: `cargo bench -- --noplot` -> failed because bench harness in `genesis-dynamics` does not accept `--noplot` (libtest), so complete workspace benchmark run aborted before completion.
- Additional measured run for target geometric benchmarks: `cargo bench -p genesis-math --bench geometry -- --noplot` (completed).

| Benchmark | Mean (ns) | 95% CI (ns) | Std Dev (ns) | Std Dev CI (ns) |
|---|---:|---:|---:|---:|
| `dense_kernel_g13_16x16` | 275.95 | [267.53, 285.60] | 46.74 | [30.83, 59.77] |
| `sparse_geo_product_16x16` | 201.02 | [196.94, 206.19] | 23.85 | [12.02, 35.76] |
| `matrix4x4_multiplication_fma` | 232.09 | [230.61, 233.70] | 7.92 | [5.77, 9.97] |

## Phase 3 — Assembly extraction (`geometric_product_x86_avx2_fma_dense`)

- Source symbol: `_ZN12genesis_math7product36geometric_product_x86_avx2_fma_dense17he6ac04dab53d43b1E` in `target/release/deps/geometry-9f9ac4388da225ee`.
- Static instruction count in function body: **156** instructions.
- Full disassembly listing:

```asm
target/release/deps/geometry-9f9ac4388da225ee:     file format elf64-x86-64


Disassembly of section .text:

0000000000201a60 <_ZN12genesis_math7product36geometric_product_x86_avx2_fma_dense17he6ac04dab53d43b1E>:
  201a60:	sub    $0x108,%rsp
  201a67:	vmovupd (%rdi),%ymm2
  201a6b:	vmovupd 0x20(%rdi),%ymm13
  201a70:	vmovupd 0x40(%rdi),%ymm12
  201a75:	vmovupd 0x60(%rdi),%ymm15
  201a7a:	vbroadcastsd (%rsi),%ymm9
  201a7f:	vxorpd %xmm8,%xmm8,%xmm8
  201a84:	vmovapd %ymm9,%ymm11
  201a89:	vfmadd213pd %ymm8,%ymm2,%ymm11
  201a8e:	vmovapd %ymm9,%ymm10
  201a93:	vfmadd213pd %ymm8,%ymm13,%ymm10
  201a98:	vmovapd %ymm9,%ymm3
  201a9c:	vfmadd213pd %ymm8,%ymm12,%ymm3
  201aa1:	vbroadcastsd 0x8(%rsi),%ymm4
  201aa7:	vfmadd213pd %ymm8,%ymm15,%ymm9
  201aac:	vmovapd -0x1cf934(%rip),%ymm14        # 32180 <_ZN10serde_json3ser9Formatter17write_char_escape10HEX_DIGITS17h096ac3aa19d490c4E+0x1fa0>
  201ab4:	vxorpd %ymm2,%ymm14,%ymm0
  201ab8:	vshufps $0x4e,%ymm0,%ymm0,%ymm1
  201abd:	vmovupd %ymm1,0x80(%rsp)
  201ac6:	vmovapd %ymm4,%ymm5
  201aca:	vfmadd213pd %ymm8,%ymm1,%ymm5
  201acf:	vshufpd $0x5,%ymm13,%ymm13,%ymm1
  201ad5:	vmovupd %ymm1,0x40(%rsp)
  201adb:	vmovddup -0x1d2bc3(%rip),%xmm7        # 2ef20 <__abi_tag+0x2ec24>
  201ae3:	vxorpd %ymm7,%ymm1,%ymm1
  201ae7:	vmovupd %ymm1,0xa0(%rsp)
  201af0:	vmovapd %ymm4,%ymm6
  201af4:	vfmadd213pd %ymm8,%ymm1,%ymm6
  201af9:	vshufpd $0x5,%ymm12,%ymm12,%ymm1
  201aff:	vmovupd %ymm1,0x20(%rsp)
  201b05:	vxorpd %ymm7,%ymm1,%ymm1
  201b09:	vmovupd %ymm1,0xe0(%rsp)
  201b12:	vmovapd %ymm4,%ymm7
  201b16:	vfmadd213pd %ymm8,%ymm1,%ymm7
  201b1b:	vshufpd $0x5,%ymm15,%ymm15,%ymm1
  201b21:	vmovupd %ymm1,0x60(%rsp)
  201b27:	vxorpd %ymm1,%ymm14,%ymm1
  201b2b:	vmovupd %ymm1,(%rsp)
  201b30:	vfmadd213pd %ymm8,%ymm1,%ymm4
  201b35:	vbroadcastsd 0x10(%rsi),%ymm1
  201b3b:	vpermpd $0x4e,%ymm0,%ymm0
  201b41:	vmovupd %ymm0,-0x20(%rsp)
  201b47:	vfmadd231pd %ymm1,%ymm0,%ymm11
  201b4c:	vpermpd $0x4e,%ymm13,%ymm0
  201b52:	vmovupd %ymm0,-0x40(%rsp)
  201b58:	vxorpd %ymm0,%ymm14,%ymm0
  201b5c:	vmovupd %ymm0,-0x60(%rsp)
  201b62:	vmovapd %ymm14,%ymm8
  201b67:	vfmadd231pd %ymm1,%ymm0,%ymm10
  201b6c:	vpermpd $0x4e,%ymm12,%ymm14
  201b72:	vxorpd %ymm8,%ymm14,%ymm0
  201b77:	vmovupd %ymm0,0xc0(%rsp)
  201b80:	vfmadd231pd %ymm1,%ymm0,%ymm3
  201b85:	vpermpd $0x4e,%ymm15,%ymm0
  201b8b:	vmovupd %ymm0,-0x80(%rsp)
  201b91:	vmovddup -0x1d2c79(%rip),%xmm8        # 2ef20 <__abi_tag+0x2ec24>
  201b99:	vxorpd %ymm0,%ymm8,%ymm8
  201b9d:	vbroadcastsd 0x20(%rsi),%ymm0
  201ba3:	vfmadd231pd %ymm1,%ymm8,%ymm9
  201ba8:	vfnmadd231pd %ymm0,%ymm13,%ymm11
  201bad:	vfmadd231pd %ymm0,%ymm2,%ymm10
  201bb2:	vfmadd231pd %ymm0,%ymm15,%ymm3
  201bb7:	vfnmadd231pd %ymm0,%ymm12,%ymm9
  201bbc:	vbroadcastsd 0x30(%rsi),%ymm0
  201bc2:	vfmadd231pd %ymm8,%ymm0,%ymm3
  201bc7:	vmovddup -0x1d2caf(%rip),%xmm1        # 2ef20 <__abi_tag+0x2ec24>
  201bcf:	vxorpd -0x40(%rsp),%ymm1,%ymm8
  201bd5:	vmovupd %ymm8,-0x40(%rsp)
  201bdb:	vfmadd231pd %ymm0,%ymm8,%ymm11
  201be0:	vmovupd -0x20(%rsp),%ymm8
  201be6:	vfmadd231pd %ymm0,%ymm8,%ymm10
  201beb:	vxorpd %ymm1,%ymm14,%ymm14
  201bef:	vfmadd231pd %ymm0,%ymm14,%ymm9
  201bf4:	vbroadcastsd 0x40(%rsi),%ymm0
  201bfa:	vfnmadd231pd %ymm0,%ymm12,%ymm11
  201bff:	vfnmadd231pd %ymm0,%ymm15,%ymm10
  201c04:	vfmadd231pd %ymm0,%ymm2,%ymm3
  201c09:	vbroadcastsd 0x50(%rsi),%ymm1
  201c0f:	vfmadd231pd %ymm0,%ymm13,%ymm9
  201c14:	vfmadd231pd %ymm14,%ymm1,%ymm11
  201c19:	vfmadd231pd -0x60(%rsp),%ymm1,%ymm9
  201c20:	vmovapd -0x1cfaa8(%rip),%ymm0        # 32180 <_ZN10serde_json3ser9Formatter17write_char_escape10HEX_DIGITS17h096ac3aa19d490c4E+0x1fa0>
  201c28:	vxorpd -0x80(%rsp),%ymm0,%ymm14
  201c2e:	vmovupd %ymm14,-0x60(%rsp)
  201c34:	vfmadd231pd %ymm1,%ymm14,%ymm10
  201c39:	vfmadd231pd %ymm1,%ymm8,%ymm3
  201c3e:	vbroadcastsd 0x60(%rsi),%ymm1
  201c44:	vfnmadd231pd %ymm15,%ymm1,%ymm11
  201c49:	vfmadd231pd %ymm12,%ymm1,%ymm10
  201c4e:	vfnmadd231pd %ymm13,%ymm1,%ymm3
  201c53:	vfmadd231pd %ymm1,%ymm2,%ymm9
  201c58:	vpermpd $0x1b,%ymm2,%ymm2
  201c5e:	vbroadcastsd 0x18(%rsi),%ymm14
  201c64:	vfmadd231pd %ymm14,%ymm2,%ymm5
  201c69:	vpermpd $0x1b,%ymm13,%ymm0
  201c6f:	vfmadd231pd %ymm14,%ymm0,%ymm6
  201c74:	vmovupd %ymm0,-0x80(%rsp)
  201c7a:	vpermpd $0x1b,%ymm12,%ymm13
  201c80:	vfmadd231pd %ymm14,%ymm13,%ymm7
  201c85:	vpermpd $0x1b,%ymm15,%ymm8
  201c8b:	vfmadd231pd %ymm14,%ymm8,%ymm4
  201c90:	vbroadcastsd 0x28(%rsi),%ymm12
  201c96:	vfmadd231pd (%rsp),%ymm12,%ymm7
  201c9c:	vmovapd -0x1cfb24(%rip),%ymm1        # 32180 <_ZN10serde_json3ser9Formatter17write_char_escape10HEX_DIGITS17h096ac3aa19d490c4E+0x1fa0>
  201ca4:	vxorpd 0x40(%rsp),%ymm1,%ymm14
  201caa:	vxorpd 0x20(%rsp),%ymm1,%ymm15
  201cb0:	vfmadd231pd %ymm12,%ymm14,%ymm5
  201cb5:	vmovupd 0x80(%rsp),%ymm1
  201cbe:	vfmadd231pd %ymm12,%ymm1,%ymm6
  201cc3:	vfmadd231pd %ymm12,%ymm15,%ymm4
  201cc8:	vbroadcastsd 0x38(%rsi),%ymm12
  201cce:	vfnmadd231pd %ymm12,%ymm0,%ymm5
  201cd3:	vfmadd231pd %ymm12,%ymm2,%ymm6
  201cd8:	vfmadd231pd %ymm12,%ymm8,%ymm7
  201cdd:	vfnmadd231pd %ymm12,%ymm13,%ymm4
  201ce2:	vbroadcastsd 0x48(%rsi),%ymm12
  201ce8:	vfmadd231pd %ymm15,%ymm12,%ymm5
  201ced:	vmovddup -0x1d2dd5(%rip),%xmm0        # 2ef20 <__abi_tag+0x2ec24>
  201cf5:	vxorpd 0x60(%rsp),%ymm0,%ymm15
  201cfb:	vfmadd231pd 0xa0(%rsp),%ymm12,%ymm4
  201d05:	vfmadd231pd %ymm12,%ymm15,%ymm6
  201d0a:	vbroadcastsd 0x58(%rsi),%ymm0
  201d10:	vfmadd231pd %ymm12,%ymm1,%ymm7
  201d15:	vfnmadd231pd %ymm0,%ymm13,%ymm5
  201d1a:	vfnmadd231pd %ymm0,%ymm8,%ymm6
  201d1f:	vmovapd %ymm2,%ymm12
  201d23:	vfmadd231pd %ymm0,%ymm2,%ymm7
  201d28:	vmovupd -0x80(%rsp),%ymm2
  201d2e:	vfmadd231pd %ymm0,%ymm2,%ymm4
  201d33:	vmovddup 0x68(%rsi),%xmm0
  201d38:	vbroadcastsd %xmm0,%ymm0
  201d3d:	vfmadd231pd %ymm15,%ymm0,%ymm5
  201d42:	vfmadd231pd 0xe0(%rsp),%ymm0,%ymm6
  201d4c:	vfmadd231pd %ymm14,%ymm0,%ymm7
  201d51:	vfmadd231pd %ymm0,%ymm1,%ymm4
  201d56:	vbroadcastsd 0x70(%rsi),%ymm0
  201d5c:	vfmadd231pd -0x60(%rsp),%ymm0,%ymm11
  201d63:	vfmadd231pd 0xc0(%rsp),%ymm0,%ymm10
  201d6d:	vfmadd231pd -0x40(%rsp),%ymm0,%ymm3
  201d74:	vfmadd231pd -0x20(%rsp),%ymm0,%ymm9
  201d7b:	vbroadcastsd 0x78(%rsi),%ymm0
  201d81:	vfnmadd231pd %ymm8,%ymm0,%ymm5
  201d86:	vfmadd231pd %ymm13,%ymm0,%ymm6
  201d8b:	vfnmadd231pd %ymm2,%ymm0,%ymm7
  201d90:	vfmadd231pd %ymm0,%ymm12,%ymm4
  201d95:	vaddpd %ymm5,%ymm11,%ymm0
  201d99:	vaddpd %ymm6,%ymm10,%ymm1
  201d9d:	vaddpd %ymm7,%ymm3,%ymm2
  201da1:	vaddpd %ymm4,%ymm9,%ymm3
  201da5:	vmovupd %ymm0,(%rdx)
  201da9:	vmovupd %ymm1,0x20(%rdx)
  201dae:	vmovupd %ymm2,0x40(%rdx)
  201db3:	vmovupd %ymm3,0x60(%rdx)
  201db8:	add    $0x108,%rsp
  201dbf:	vzeroupper
  201dc2:	ret

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
```

## Phase 4 — Instruction mix analysis

- FMA instructions: 64 (41.03%).
- Memory ops (detected from memory operands): loads 43 (27.56%), stores 19 (12.18%).
- Shuffle/permutation instructions: 12 (7.69%).
- Scalar/control instructions: 4 (2.56%).
- Measured mix indicates arithmetic-heavy kernel with substantial memory traffic for temporaries (stack scratch) and moderate shuffle usage.

## Phase 5 — Register pressure

- YMM registers used: 16 (`ymm0, ymm1, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9`).
- Stack memory references: 31 total (`stack loads=16`, `stack stores=15`).
- **Spills detected:** yes. Multiple `vmovupd` stores/loads to `[rsp + offset]` appear in the hot function body, indicating register-pressure-driven spilling/scratch usage.

## Phase 6 — Port utilization estimate (requested AVX2 mapping)

- Mapping used (per mission): FMA→ports 0/1, loads→ports 2/3, shuffle→port 5.
- FMA pressure estimate: 64/2 = 32.0 cycles.
- Load pressure estimate: 43/2 = 21.5 cycles.
- Shuffle pressure estimate: 12/1 = 12.0 cycles.
- Dominant port group by this model: **ports 0/1 (FMA)** with 32.0 cycle-equivalent pressure.

## Phase 7 — Memory traffic

- Loads from `a` (`%rdi`): 4 memory references (vector block loads).
- Loads from `b` (`%rsi`): 16 memory references (broadcast scalar coefficients).
- Stores to result (`%rdx`): 4 memory references (4 vector stores = 16 coefficients).
- Stack scratch traffic (`%rsp`): 31 memory references.
- The input/output footprint (a+b+result) is small enough for L1 residency, but stack scratch traffic is non-trivial and present in the core compute region.

## Phase 8 — Latency chains

- Long accumulator chains are present (e.g., repeated `vfmadd231pd`/`vfnmadd231pd` updating `%ymm11`, `%ymm10`, `%ymm3`, `%ymm9`, `%ymm5`, `%ymm6`, `%ymm7`, `%ymm4`).
- Accumulators are interleaved across several registers, but each accumulator still has multi-instruction dependent sequences, which creates latency exposure if issue overlap is insufficient.

## Phase 9 — Theoretical throughput model

Using mission formula:
```text
cycles_per_iteration = max(FMA_count/FMA_throughput, load_count/load_ports, shuffle_count/shuffle_port)
```
- `max(64/2, 43/2, 12/1) = 32.0` cycles.
- At 2.80 GHz nominal: `32.0 / 2.8 ≈ 11.43 ns` theoretical lower bound.
- Measured dense benchmark mean: `275.95 ns`.
- Gap (measured / model): `24.1x` above simple throughput floor.

## Phase 10 — Bottleneck ranking (data-driven)

1. **Register pressure + stack spills/scratch traffic** (31 stack refs, including 15 stores and 16 loads).
2. **FMA execution pressure on ports 0/1** (largest modeled port demand).
3. **FMA dependency chains (latency chains on accumulators)**, limiting ideal overlap.
4. **Shuffle/permutation pressure** (moderate, non-dominant vs FMA).
5. **Input/output memory bandwidth** (small I/O footprint; less dominant than internal stack traffic).

## Phase 11 — Summary

- Full forensic baseline has been captured from measured benchmark and disassembly data, with no optimization changes applied.
