# ============================================================
# TASK 02 - Customer Segmentation using K-Means Clustering
# SkillCraft Technology - Data Science Internship
# ============================================================

# STEP 1: Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  TASK 02: Customer Segmentation - K-Means Clustering")
print("=" * 60)

# ============================================================
# STEP 2: Load the Dataset
# Download "Mall_Customers.csv" from the task link and place
# it in the same folder as this script.
# ============================================================

try:
    df = pd.read_csv("Mall_Customers.csv")
    print(f"\n✅ Dataset loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("\n⚠️  Dataset not found. Creating sample dataset for demonstration...")
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'CustomerID': range(1, n+1),
        'Genre': np.random.choice(['Male', 'Female'], n),
        'Age': np.random.randint(18, 70, n),
        'Annual Income (k$)': np.random.randint(15, 140, n),
        'Spending Score (1-100)': np.random.randint(1, 100, n)
    })
    print(f"✅ Sample dataset created! Shape: {df.shape}")

# ============================================================
# STEP 3: Explore the Dataset
# ============================================================

print("\n📊 DATASET OVERVIEW:")
print("-" * 40)
print(df.head())
print(f"\nColumns: {list(df.columns)}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Statistics:\n{df.describe()}")

# ============================================================
# STEP 4: Select Features for Clustering
# We use Annual Income and Spending Score (most common)
# ============================================================

# Auto-detect column names
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

income_col  = find_col(df, ['Annual Income (k$)', 'Annual_Income', 'income', 'Income'])
spending_col = find_col(df, ['Spending Score (1-100)', 'Spending_Score', 'spending_score', 'SpendingScore'])
age_col     = find_col(df, ['Age', 'age'])

print(f"\n🔍 Detected Columns:")
print(f"   Annual Income  : {income_col}")
print(f"   Spending Score : {spending_col}")
print(f"   Age            : {age_col}")

# Use Income + Spending Score as primary features
if income_col and spending_col:
    feature_cols = [income_col, spending_col]
elif age_col and income_col:
    feature_cols = [age_col, income_col]
else:
    feature_cols = df.select_dtypes(include=[np.number]).columns[:2].tolist()

print(f"\n✅ Features selected for clustering: {feature_cols}")

X = df[feature_cols].dropna()

# ============================================================
# STEP 5: Feature Scaling
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# STEP 6: Find Optimal K using Elbow Method + Silhouette Score
# ============================================================

print("\n🔍 Finding optimal number of clusters (K)...")
inertia_values    = []
silhouette_values = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertia_values.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouette_values.append(sil)
    print(f"   K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil:.4f}")

best_k = K_range[np.argmax(silhouette_values)]
print(f"\n✅ Optimal K (by Silhouette Score): {best_k}")

# ============================================================
# STEP 7: Train Final K-Means Model
# ============================================================

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean = df[feature_cols].dropna().copy()
df_clean['Cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"\n✅ K-Means model trained with K={best_k}!")
print(f"\n📊 Cluster Distribution:")
print(df_clean['Cluster'].value_counts().sort_index())

# Cluster Centers (inverse scaled)
centers = scaler.inverse_transform(kmeans_final.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=feature_cols)
centers_df.index.name = 'Cluster'
print(f"\n📍 Cluster Centers (original scale):")
print(centers_df)

# ============================================================
# STEP 8: Analyze Each Cluster
# ============================================================

print("\n📊 CLUSTER ANALYSIS:")
print("-" * 50)
cluster_stats = df_clean.groupby('Cluster')[feature_cols].mean()
cluster_stats['Count'] = df_clean.groupby('Cluster').size()

for cluster_id in sorted(df_clean['Cluster'].unique()):
    row = cluster_stats.loc[cluster_id]
    print(f"\n  Cluster {cluster_id} ({int(row['Count'])} customers):")
    for col in feature_cols:
        print(f"    Avg {col:30s}: {row[col]:.2f}")

# ============================================================
# STEP 9: Visualizations
# ============================================================

colors = plt.cm.Set1(np.linspace(0, 1, best_k))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Task 02: Customer Segmentation - K-Means Clustering\nSkillCraft Technology',
             fontsize=14, fontweight='bold')

# Plot 1: Elbow Method
axes[0, 0].plot(list(K_range), inertia_values, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (K)')
axes[0, 0].set_ylabel('Inertia (WCSS)')
axes[0, 0].set_title('Elbow Method - Finding Optimal K')
axes[0, 0].axvline(x=best_k, color='red', linestyle='--', label=f'Optimal K={best_k}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Silhouette Scores
axes[0, 1].plot(list(K_range), silhouette_values, 'rs-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Clusters (K)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Score per K')
axes[0, 1].axvline(x=best_k, color='blue', linestyle='--', label=f'Best K={best_k}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Customer Clusters Scatter Plot
cluster_labels = df_clean['Cluster'].values
scatter = axes[1, 0].scatter(
    df_clean[feature_cols[0]], 
    df_clean[feature_cols[1]],
    c=cluster_labels, cmap='Set1', 
    alpha=0.7, s=60, edgecolors='k', linewidths=0.3
)
# Plot cluster centers
axes[1, 0].scatter(
    centers_df[feature_cols[0]], 
    centers_df[feature_cols[1]],
    c='black', s=200, marker='X', zorder=5, label='Cluster Centers'
)
axes[1, 0].set_xlabel(feature_cols[0])
axes[1, 0].set_ylabel(feature_cols[1])
axes[1, 0].set_title(f'Customer Segments (K={best_k})')
axes[1, 0].legend()
plt.colorbar(scatter, ax=axes[1, 0])

# Plot 4: Cluster Size Bar Chart
cluster_counts = df_clean['Cluster'].value_counts().sort_index()
bar_colors = [plt.cm.Set1(i / best_k) for i in range(best_k)]
axes[1, 1].bar([f'Cluster {i}' for i in cluster_counts.index], cluster_counts.values,
               color=bar_colors, edgecolor='black')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Number of Customers')
axes[1, 1].set_title('Customer Count per Cluster')
for i, v in enumerate(cluster_counts.values):
    axes[1, 1].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('task02_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot saved as 'task02_results.png'")

# ============================================================
# STEP 10: Predict Cluster for a New Customer
# ============================================================

print("\n🛒 SAMPLE PREDICTION FOR NEW CUSTOMER:")
print("-" * 40)

new_customer = pd.DataFrame({
    feature_cols[0]: [75],   # e.g., Annual Income $75k
    feature_cols[1]: [60],   # e.g., Spending Score 60
})

new_scaled = scaler.transform(new_customer)
predicted_cluster = kmeans_final.predict(new_scaled)[0]
print(f"  Input : {dict(zip(feature_cols, new_customer.values[0]))}")
print(f"  Assigned to Cluster: {predicted_cluster}")

print("\n" + "=" * 60)
print("  ✅ TASK 02 COMPLETE!")
print("=" * 60)
