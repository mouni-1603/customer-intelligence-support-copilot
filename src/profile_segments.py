
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# 1. Load RFM data
rfm = pd.read_csv(OUTPUTS / "rfm_table.csv")

features = ["recency", "frequency", "monetary"]
X = rfm[features]

# 2. Standardise RFM features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit final K-Means model with K=4
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = kmeans.fit_predict(X_scaled)

# 4. Profile the four segments
profile = (
    rfm.groupby("segment_id")
    .agg(
        customer_count=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        median_recency=("recency", "median"),
        median_frequency=("frequency", "median"),
        median_monetary=("monetary", "median")
    )
    .round(2)
)

print("\n" + "=" * 70)
print("CUSTOMER SEGMENT PROFILES")
print("=" * 70)
print(profile.to_string())

# 5. Identify Premium segment:
# The segment with the highest average monetary value.
premium_segment = profile["avg_monetary"].idxmax()

print("\nPremium segment ID:", premium_segment)


# 6. Assign business-friendly names based on the observed profiles.
segment_names = {
    0: "Occasional",
    1: "Premium",
    2: "Loyal Active",
    3: "Dormant"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"] == premium_segment

# 7. PCA with two components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_

print("\n" + "=" * 70)
print("PCA RESULTS")
print("=" * 70)
print(f"PC1 explained variance: {explained[0]:.4f}")
print(f"PC2 explained variance: {explained[1]:.4f}")
print(f"Total explained variance: {explained.sum():.4f}")

# 8. Save final customer segmentation output
final_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium"
]

customer_segments = rfm[final_columns].copy()

output_file = OUTPUTS / "customer_segments.csv"
customer_segments.to_csv(output_file, index=False)

# 9. Validation checks
assert len(customer_segments) == 1175
assert customer_segments["segment_id"].nunique() == 4
assert customer_segments["is_premium"].sum() == (
    profile.loc[premium_segment, "customer_count"]
)

print("\n" + "=" * 70)
print("FINAL SEGMENT COUNTS")
print("=" * 70)
print(
    customer_segments.groupby(
        ["segment_id", "segment_name", "is_premium"]
    ).size().to_string()
)

print("\nSaved:", output_file)
print("Total customers:", len(customer_segments))
print("Number of segments:", customer_segments["segment_id"].nunique())
print("\nSEGMENT PROFILING COMPLETED SUCCESSFULLY")
