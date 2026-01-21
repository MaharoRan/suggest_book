# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("BOOK CLASSIFICATION - HYBRID SEMANTIC APPROACH")
print("="*80)

# =============================================================================
# STEP 1: LOAD AND PREPARE DATA
# =============================================================================
print("\n[STEP 1/6] Loading and cleaning dataset...")
model = SentenceTransformer("all-MiniLM-L6-v2")

dataset = pd.read_csv('Book_Dataset_1.csv', sep=',', encoding='latin1')
dataset = dataset.drop_duplicates()

# Remove invalid categories
dataset = dataset[
    (dataset['Category'].str.strip() != '') &
    (dataset['Category'].str.lower() != 'default') &
    (dataset['Category'].str.lower() != 'add a comment')
]

print(f"[OK] {len(dataset)} books loaded")

# =============================================================================
# STEP 2: SMART CATEGORY GROUPING
# =============================================================================
print("\n[STEP 2/6] Grouping similar categories...")

def clean_text(text):
    if pd.isna(text):
        return ''
    return str(text).lower().strip()

dataset['category_clean'] = dataset['Category'].apply(clean_text)

# Category mapping
category_mapping = {
    # Fiction genres
    'fiction': 'fiction',
    'historical fiction': 'fiction',
    'womens fiction': 'fiction',
    'adult fiction': 'fiction',
    'contemporary fiction': 'fiction',
    
    # Sci-fi & Fantasy
    'fantasy': 'fantasy_scifi',
    'science fiction': 'fantasy_scifi',
    'paranormal': 'fantasy_scifi',
    
    # Mystery & Thriller
    'mystery': 'mystery_thriller',
    'thriller': 'mystery_thriller',
    'crime': 'mystery_thriller',
    'suspense': 'mystery_thriller',
    
    # Romance genres
    'romance': 'romance',
    'historical romance': 'romance',
    
    # Young readers
    'young adult': 'young_adult',
    'childrens': 'childrens',
    
    # Non-fiction categories
    'nonfiction': 'nonfiction',
    'biography': 'nonfiction',
    'autobiography': 'nonfiction',
    'history': 'nonfiction',
    'philosophy': 'nonfiction',
    
    # Arts & Literature
    'poetry': 'poetry_literature',
    'classics': 'poetry_literature',
    'novels': 'poetry_literature',
    
    # Visual arts
    'sequential art': 'sequential_art',
    'comics': 'sequential_art',
    'graphic novels': 'sequential_art',
    
    # Practical
    'food and drink': 'lifestyle',
    'cookbooks': 'lifestyle',
    'health': 'lifestyle',
    'self help': 'lifestyle',
    'parenting': 'lifestyle',
    
    # Academic
    'science': 'academic',
    'business': 'academic',
    'psychology': 'academic',
    'politics': 'academic',
    'academic': 'academic',
    
    # Other
    'horror': 'horror',
    'humor': 'humor',
    'travel': 'travel',
    'music': 'arts',
    'art': 'arts',
    'cultural': 'arts',
    'spirituality': 'spirituality',
    'religion': 'spirituality',
    'christian': 'spirituality',
}

dataset['category_grouped'] = dataset['category_clean'].map(
    lambda x: category_mapping.get(x, 'other')
)

# Keep only categories with enough books
min_books = 20
grouped_counts = dataset['category_grouped'].value_counts()
valid_categories = grouped_counts[grouped_counts >= min_books].index
dataset = dataset[dataset['category_grouped'].isin(valid_categories)]
dataset = dataset.reset_index(drop=True)

print(f"[OK] {len(valid_categories)} final categories with >={min_books} books")
print(f"[OK] {len(dataset)} books retained")

# =============================================================================
# STEP 3: GENERATE SEMANTIC EMBEDDINGS
# =============================================================================
print("\n[STEP 3/6] Generating semantic embeddings...")

dataset['title_clean'] = dataset['Title'].apply(clean_text)
dataset['description_clean'] = dataset['Book_Description'].apply(clean_text)
dataset['text_for_embedding'] = (
    dataset['title_clean'] + ' ' + dataset['description_clean']
)

embeddings = model.encode(
    dataset['text_for_embedding'].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

X_all = embeddings.cpu().numpy()
y_all = dataset['category_grouped']

# Normalize
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

print(f"[OK] Embeddings shape: {X_all.shape}")

# =============================================================================
# STEP 4: TRAIN/TEST SPLIT & CREATE CATEGORY PROFILES
# =============================================================================
print("\n[STEP 4/6] Splitting data and creating category profiles...")

indices = np.arange(len(dataset))
idx_train, idx_test = train_test_split(
    indices, 
    test_size=0.25,
    random_state=42, 
    stratify=y_all
)

# Create category profiles (centroids)
category_profiles = {}
for category in dataset.iloc[idx_train]['category_grouped'].unique():
    category_mask = dataset.iloc[idx_train]['category_grouped'] == category
    train_indices_for_cat = idx_train[category_mask[idx_train]]
    category_embeddings = X_all[train_indices_for_cat]
    category_profiles[category] = np.mean(category_embeddings, axis=0)

X_train = X_all[idx_train]
y_train = dataset.iloc[idx_train]['category_grouped'].values
y_test = dataset.iloc[idx_test]['category_grouped'].values

print(f"[OK] Train: {len(idx_train)} | Test: {len(idx_test)}")
print(f"[OK] {len(category_profiles)} category profiles created")

# =============================================================================
# STEP 5: HYBRID PREDICTION (CENTROID + K-NN)
# =============================================================================
print("\n[STEP 5/6] Running hybrid prediction...")

k = 5  # Number of neighbors
alpha = 0.6  # Weight for centroid
beta = 0.4   # Weight for K-NN

y_pred_hybrid = []

for idx in idx_test:
    query_embedding = X_all[idx].reshape(1, -1)
    
    # Centroid scores
    centroid_scores = {}
    for category, profile in category_profiles.items():
        similarity = cosine_similarity(query_embedding, profile.reshape(1, -1))[0][0]
        centroid_scores[category] = similarity
    
    # K-NN scores
    similarities = cosine_similarity(query_embedding, X_train)[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    neighbors_categories = y_train[top_k_indices]
    
    knn_scores = {}
    for cat in category_profiles.keys():
        knn_scores[cat] = (neighbors_categories == cat).sum() / k
    
    # Hybrid combination
    hybrid_scores = {}
    for cat in category_profiles.keys():
        hybrid_scores[cat] = alpha * centroid_scores[cat] + beta * knn_scores[cat]
    
    predicted_category = max(hybrid_scores.items(), key=lambda x: x[1])[0]
    y_pred_hybrid.append(predicted_category)

# =============================================================================
# STEP 6: EVALUATION
# =============================================================================
print("\n" + "="*80)
print("RESULTS")
print("="*80)

accuracy = accuracy_score(y_test, y_pred_hybrid)
precision = precision_score(y_test, y_pred_hybrid, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_hybrid, average='weighted', zero_division=0)

print(f"\n[STEP 6/6] Hybrid Method (alpha={alpha}, beta={beta}, K={k})")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   F1-Score:  {f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_hybrid, zero_division=0))

print("\n" + "="*80)
print("[SUCCESS] CLASSIFICATION COMPLETE")
print("="*80)
