
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# --------------------------------------------------
# 1. LOAD THE RFM TABLE
# --------------------------------------------------

rfm = pd.read_csv(OUTPUTS / "rfm_table.csv")

features = ["recency", "frequency", "monetary"]
X = rfm[features]

# --------------------------------------------------
# 2. STANDARDISE THE RFM FEATURES
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nRFM features standardised successfully.")
print("Number of customers:", X_scaled.shape[0])
print("Number of features:", X_scaled.shape[1])

# --------------------------------------------------
# 3. TEST K FROM 2 TO 8
# --------------------------------------------------

results = []

for k in range(2, 9):
    print(f"\nTraining KMeans for K={k}...")

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette Score": silhouette
    })

# --------------------------------------------------
# 4. DISPLAY AND SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 65)
print("K-MEANS CLUSTER EVALUATION")
print("=" * 65)

print(results_df.round(3).to_string(index=False))

results_path = OUTPUTS / "kmeans_evaluation.csv"
results_df.to_csv(results_path, index=False)

print("\nEvaluation results saved to:", results_path)

# --------------------------------------------------
# 5. REFERENCE CHECKPOINTS
# --------------------------------------------------

reference = {
    2: (1820.8, 0.465),
    3: (1268.2, 0.445),
    4: (743.1, 0.526),
    5: (617.4, 0.481),
    6: (503.2, 0.482),
    7: (408.5, 0.416),
    8: (366.5, 0.412)
}

print("\n" + "=" * 65)
print("REFERENCE COMPARISON")
print("=" * 65)

for _, row in results_df.iterrows():
    k = int(row["K"])
    expected_inertia, expected_silhouette = reference[k]

    inertia_difference = (
        abs(row["Inertia"] - expected_inertia) / expected_inertia
    )

    silhouette_difference = abs(
        row["Silhouette Score"] - expected_silhouette
    )

    inertia_pass = inertia_difference <= 0.05
    silhouette_pass = silhouette_difference <= 0.05

    print(
        f"K={k}: "
        f"Inertia={row['Inertia']:.2f} "
        f"({'PASS' if inertia_pass else 'CHECK'}), "
        f"Silhouette={row['Silhouette Score']:.3f} "
        f"({'PASS' if silhouette_pass else 'CHECK'})"
    )

# --------------------------------------------------
# 6. IDENTIFY THE BEST SILHOUETTE SCORE
# --------------------------------------------------

best_row = results_df.loc[
    results_df["Silhouette Score"].idxmax()
]

print("\nHighest silhouette score:")
print("K:", int(best_row["K"]))
print("Score:", round(best_row["Silhouette Score"], 3))

print("\nK-MEANS EVALUATION COMPLETED")
