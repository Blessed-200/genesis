"""
Fase 0 - script de scoring honesto.

Uso:
    python3 score_results.py phase0_output.csv

Espera un CSV con columnas: node_id,final_phase,label
(el mismo orden/labels que digits_16d.csv, para que sea comparable)

Imprime el ARI/NMI del pipeline real de GENESIS (E) al lado de los
baselines ya establecidos (A, B, C, D) para que la comparación sea
de un vistazo. No cambies los numeros de A-D salvo que vuelvas a
correr validate_hypothesis.py con otra semilla/dataset - deben ser
la misma corrida para que la comparacion sea justa.
"""
import csv
import sys
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering

# Resultados de referencia ya medidos en esta conversacion (digits, N=600, seed=42).
# Si cambias el dataset o el N, vuelve a correr validate_hypothesis.py para
# regenerar esta tabla - comparar contra numeros de otra corrida no es valido.
BASELINES = {
    "A) KMeans en 64d crudo":                    dict(ari=0.528, nmi=0.682),
    "B) KMeans en proyeccion 16d":                dict(ari=0.372, nmi=0.564),
    "C) Spectral clustering en grafo kNN (sin dinamica)": dict(ari=0.542, nmi=0.728),
    "D) Kuramoto ingenuo (sin VFE) + spectral":   dict(ari=0.169, nmi=0.441),
}


def score_genesis_output(path: str):
    thetas, labels = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            thetas.append(float(row["final_phase"]))
            labels.append(int(row["label"]))
    theta = np.array(thetas)
    y = np.array(labels)

    n = len(theta)
    sim = (np.cos(theta[:, None] - theta[None, :]) + 1) / 2
    labels_pred = SpectralClustering(
        n_clusters=len(set(y)), affinity="precomputed", random_state=0, assign_labels="kmeans"
    ).fit_predict(sim)

    ari = adjusted_rand_score(y, labels_pred)
    nmi = normalized_mutual_info_score(y, labels_pred)
    return ari, nmi


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    ari_e, nmi_e = score_genesis_output(sys.argv[1])

    print("\n=== Fase 0: comparacion honesta ===\n")
    for name, vals in BASELINES.items():
        print(f"{name:<50} ARI={vals['ari']:.3f}  NMI={vals['nmi']:.3f}")
    print(f"{'E) Pipeline real GENESIS (HNSW+Kuramoto+VFE)':<50} ARI={ari_e:.3f}  NMI={nmi_e:.3f}")

    best_baseline_ari = max(v["ari"] for v in BASELINES.values())
    print()
    if ari_e > best_baseline_ari:
        print(f"-> GENESIS supera al mejor baseline ({best_baseline_ari:.3f}). Senal real: vale la pena seguir invirtiendo aqui.")
    else:
        print(f"-> GENESIS NO supera al mejor baseline ({best_baseline_ari:.3f}). Resultado negativo honesto: revisar la hipotesis antes de seguir construyendo encima.")


if __name__ == "__main__":
    main()
