# Exploratory data analysis (EDA) on a cancer dataset
# Loading the files and exploring the data with pandas
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# Load the data
####################################################
data = pd.read_csv('/Users/amelialuongo/Desktop/comp bme/Module-4-Cancer/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)
metadata_df = pd.read_csv('/Users/amelialuongo/Desktop/comp bme/Module-4-Cancer/data/TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
print(data.head())

# %%
# Explore the data
####################################################
print(data.shape)
print(data.info())
print(data.describe())

# %%
# Explore the metadata
####################################################
print(metadata_df.info())
print(metadata_df.describe())

# %%
# Subset the data for a specific cancer type
####################################################
cancer_type = ['CESC', 'OV', 'BRCA', 'UCEC', 'UCS']

# From metadata, get the rows where "cancer_type" is equal to the specified cancer type
# Then grab the index of this subset (these are the sample IDs)
cancer_samples = metadata_df[metadata_df['cancer_type'].isin(cancer_type)].index
print(cancer_samples)
# Subset the main data to include only these samples
# When you want a subset of columns, you can pass a list of column names to the data frame in []
cancer_data = data[cancer_samples]

# %%
# Subset by index (genes)
####################################################

# Load gene list from Menyhart hallmarks file and extract metastasis row
hallmarks = {}
with open('/Users/amelialuongo/Desktop/comp bme/Module-4-Cancer/data/Menyhart_JPA_CancerHallmarks_core.csv') as f:
    for line in f:
        parts = line.strip().strip('"').split('\t')
        hallmarks[parts[0]] = parts[1:]

desired_gene_list = hallmarks['TISSUE INVASION AND METASTASIS']

gene_list = [gene for gene in desired_gene_list if gene in cancer_data.index]


# .loc[] is the method to subset by index labels
# .iloc[] will subset by index position (integer location) instead
cancer_gene_data = cancer_data.loc[gene_list]
print(cancer_gene_data.head())

print(gene_list)
print(f"\nTotal: {len(gene_list)} genes")




# %%
# Basic statistics on the subsetted data
####################################################
print(cancer_gene_data.describe())
print(cancer_gene_data.var(axis=1))  # Variance of each gene across samples
# Mean expression of each gene across samples
print(cancer_gene_data.mean(axis=1))
# Median expression of each gene across samples
print(cancer_gene_data.median(axis=1))

# %%
# Explore categorical variables in metadata
####################################################
# groupby allows you to group on a specific column in the dataset,
# and then print out summary stats or counts for other columns within those groups
print(metadata_df.groupby('cancer_type')["gender"].value_counts())

# Explore average age at diagnosis by cancer type
metadata_df['age_at_diagnosis'] = pd.to_numeric(
    metadata_df['age_at_diagnosis'], errors='coerce')
print(metadata_df.groupby(
    'cancer_type')["age_at_diagnosis"].mean())
# %%
# Merging datasets
####################################################
# Merge the subsetted expression data with metadata for BRCA samples,
# so rows are samples and columns include gene expression for EGFR and MYC and metadata
cancer_metadata = metadata_df.loc[cancer_samples]
cancer_merged = cancer_gene_data.T.merge(
    cancer_metadata, left_index=True, right_index=True)
print(cancer_merged.head())

# %%
# Plotting - Gene expression across cancer types
####################################################

# Boxplot of a key metastasis gene across cancer types
sns.boxplot(data=cancer_merged, x='cancer_type', y='SNAI1')
plt.title('SNAI1 (EMT driver) Expression Across Gynecological Cancers')
plt.xlabel('Cancer Type')
plt.ylabel('log2 TPM Expression')
plt.show()

# Boxplot of key metastasis gene by contraceptive use
# Filter out unknown to focus on samples with actual data
known_hormone = cancer_merged[cancer_merged['history_hormonal_contraceptives_use'].isin(['Never Used', 'Former User', 'Current User'])]

sns.boxplot(data=known_hormone, x='history_hormonal_contraceptives_use', y='SNAI1')
plt.title('SNAI1 Expression by Hormonal Contraceptive Use')
plt.xlabel('Contraceptive Use History')
plt.ylabel('log2 TPM Expression')
plt.show()

# Boxplot comparing contraceptive use AND cancer type together
sns.boxplot(data=known_hormone, x='cancer_type', y='SNAI1', hue='history_hormonal_contraceptives_use')
plt.title('SNAI1 Expression by Cancer Type and Contraceptive Use')
plt.xlabel('Cancer Type')
plt.ylabel('log2 TPM Expression')
plt.legend(title='Contraceptive Use', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Prepare data for PCA
####################################################
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Transpose so rows=samples, columns=genes, then scale
X = cancer_gene_data.T
X_scaled = StandardScaler().fit_transform(X)

# %%
# Run PCA and plot - colored by cancer type
####################################################
X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

cancer_labels = metadata_df.loc[X.index, 'cancer_type']

fig, ax = plt.subplots(figsize=(8, 6))
for ctype in cancer_labels.unique():
    mask = cancer_labels == ctype
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=ctype, alpha=0.7, s=30)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA - Colored by Cancer Type')
ax.legend(title='Cancer type')
plt.show()

# %%
# PCA - colored by hormonal contraceptive use
####################################################
hormone_labels = metadata_df.loc[X.index, 'history_hormonal_contraceptives_use']
hormone_labels = hormone_labels.fillna('Unknown').replace(['[Not Available]', '[Unknown]'], 'Unknown')

fig, ax = plt.subplots(figsize=(8, 6))
for val in hormone_labels.unique():
    mask = hormone_labels == val
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=val, alpha=0.7, s=30)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA - Colored by Hormonal Contraceptive Use')
ax.legend(title='Contraceptive use')
plt.show()

# %%
# K-means clustering on PCA
####################################################
from sklearn.cluster import KMeans

km = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_pca)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('K-means Clusters on PCA')
plt.colorbar(scatter, label='Cluster')
plt.show()

# %%
# UMAP
####################################################
import umap

X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
for ctype in cancer_labels.unique():
    mask = cancer_labels == ctype
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1], label=ctype, alpha=0.7, s=30)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('UMAP - Colored by Cancer Type')
ax.legend(title='Cancer type')
plt.show()
