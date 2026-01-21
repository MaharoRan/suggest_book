import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

print("Chargement du modèle SentenceTransformer (SBERT)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Charger le dataset
dataset = pd.read_csv('Book_Dataset_1.csv', sep=',', encoding='latin1')

dataset = dataset.drop_duplicates()

# Supprimer les livres avec catégorie "Default", "add a comment" ou vide
dataset = dataset[
    (dataset['Category'].str.strip() != '') &
    (dataset['Category'].str.lower() != 'default')&
    (dataset['Category'].str.lower() != 'add a comment')
]

print(dataset.Title.count(), "livres après nettoyage des catégories.")

# NOUVEAU: Filtrer les catégories avec trop peu d'exemples
def clean_text(text):
    if pd.isna(text):
        return ''
    return str(text).lower().strip()

dataset['category_clean'] = dataset['Category'].apply(clean_text)

# Analyser la distribution
category_counts = dataset['category_clean'].value_counts()
print(f"\nNombre de catégories uniques: {len(category_counts)}")

# Garder seulement les catégories avec au moins 10 livres
min_books_per_category = 10
valid_categories = category_counts[category_counts >= min_books_per_category].index
dataset = dataset[dataset['category_clean'].isin(valid_categories)]

print(f"Filtrage: gardé {len(valid_categories)} catégories avec ≥ {min_books_per_category} livres")
print(f"Total de livres après filtrage: {len(dataset)}")

print("\nDistribution des catégories retenues:")
for category, count in dataset['category_clean'].value_counts().head(10).items():
    print(f"  - {category}: {count} livres")

# Nettoyer et préparer le texte pour embedding
dataset = dataset.reset_index(drop=True)  # Réinitialiser les index

def clean_text(text):
    if pd.isna(text):
        return ''
    return str(text).lower().strip()
    if pd.isna(text):
        return ''
    return str(text).lower().strip()

dataset['title_clean'] = dataset['Title'].apply(clean_text)
dataset['category_clean'] = dataset['Category'].apply(clean_text)
dataset['description_clean'] = dataset['Book_Description'].apply(clean_text)

# Regrouper les champs pour l'embedding
dataset['text_for_embedding'] = (
    dataset['title_clean'] + ' ' +
    dataset['category_clean'] + ' ' +
    dataset['description_clean']
)

# Créer embeddings avec SentenceTransformer
print("Génération des embeddings SentenceTransformer (384 dimensions)...")
embeddings = model.encode(
    dataset['text_for_embedding'].tolist(),
    convert_to_tensor=True
)

# Convertir en numpy array pour sklearn
X = embeddings.cpu().numpy()
y = dataset['category_clean']

print(f"Forme de X: {X.shape}")
print(f"Dimension des embeddings: {X.shape[1]}")

# Normaliser les données
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Modèle 1: Random Forest
print("\n" + "="*80)
print("ÉVALUATION: Random Forest Classifier")
print("="*80)
model_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
print(f"\n📊 Résultats Random Forest:")
print(f"   Accuracy: {rf_accuracy:.4f} | Precision: {rf_precision:.4f} | F1: {rf_f1:.4f}")

# Rapport détaillé (top/bottom classes)
print("\nTop 5 catégories les mieux prédites:")
report = classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)
class_f1 = [(k, v['f1-score']) for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']]
class_f1_sorted = sorted(class_f1, key=lambda x: x[1], reverse=True)
for cat, f1 in class_f1_sorted[:5]:
    print(f"   {cat[:30]:30} F1: {f1:.3f}")

# =============================================================================
# APPROCHE SÉMANTIQUE PURE (sans ML classique)
# =============================================================================

print("\n" + "="*80)
print("ÉVALUATION: Approche Sémantique Pure (Cosine Similarity)")
print("="*80)

# Créer les profils de catégories AVANT de séparer train/test
X_all = embeddings.cpu().numpy()
X_all = scaler.fit_transform(X_all)
y_all = dataset['category_clean']

# Créer des indices pour train/test
from sklearn.model_selection import train_test_split
indices = np.arange(len(dataset))
idx_train, idx_test = train_test_split(indices, test_size=0.3, random_state=42, stratify=y_all)

# Créer les profils de catégories basés UNIQUEMENT sur le train set
category_profiles_semantic = {}
for category in dataset.iloc[idx_train]['category_clean'].unique():
    category_mask = dataset.iloc[idx_train]['category_clean'] == category
    train_indices_for_cat = idx_train[category_mask[idx_train]]
    category_embeddings = X_all[train_indices_for_cat]
    category_profiles_semantic[category] = np.mean(category_embeddings, axis=0)

# Prédire sur le test set
y_pred_semantic = []
for idx in idx_test:
    query_embedding = X_all[idx].reshape(1, -1)
    
    # Calculer similarité avec chaque profil
    similarities = {}
    for category, profile in category_profiles_semantic.items():
        similarity = cosine_similarity(query_embedding, profile.reshape(1, -1))[0][0]
        similarities[category] = similarity
    
    # Prédire la catégorie la plus similaire
    predicted_category = max(similarities.items(), key=lambda x: x[1])[0]
    y_pred_semantic.append(predicted_category)

# Évaluer
y_test_semantic = dataset.iloc[idx_test]['category_clean'].values
sem_accuracy = accuracy_score(y_test_semantic, y_pred_semantic)
sem_precision = precision_score(y_test_semantic, y_pred_semantic, average='weighted', zero_division=0)
sem_f1 = f1_score(y_test_semantic, y_pred_semantic, average='weighted', zero_division=0)

print(f"\n📊 Résultats Approche Sémantique:")
print(f"   Accuracy: {sem_accuracy:.4f} | Precision: {sem_precision:.4f} | F1: {sem_f1:.4f}")

print("\n" + "="*80)
print("COMPARAISON DES APPROCHES")
print("="*80)
print(f"{'Méthode':<30} {'Accuracy':<12} {'Precision':<12} {'F1-Score':<12}")
print("-"*80)
print(f"{'Random Forest':<30} {rf_accuracy:<12.4f} {rf_precision:<12.4f} {rf_f1:<12.4f}")
print(f"{'Sémantique (Cosine)':<30} {sem_accuracy:<12.4f} {sem_precision:<12.4f} {sem_f1:<12.4f}")

improvement = ((sem_accuracy - rf_accuracy) / rf_accuracy) * 100 if rf_accuracy > 0 else 0
print(f"\n{'Amélioration sémantique:':<30} {improvement:+.2f}%")

# =============================================================================
# ANALYSE SÉMANTIQUE CONTEXTUELLE
# =============================================================================

print("\n" + "="*80)
print("ANALYSE SÉMANTIQUE ET CONTEXTUELLE")
print("="*80)

# 1. Créer des profils de catégories basés sur les embeddings moyens (sur tout le dataset)
print("\n1. Création des profils sémantiques par catégorie...")
category_profiles = {}
for category in dataset['category_clean'].unique():
    category_mask = dataset['category_clean'] == category
    category_indices = np.where(category_mask)[0]
    category_embeddings = X_all[category_indices]
    # Profil = centroïde des embeddings de cette catégorie
    category_profiles[category] = np.mean(category_embeddings, axis=0)

print(f"   → {len(category_profiles)} profils de catégories créés")

# 2. Fonction de recherche sémantique par similarité cosine
def find_similar_books_by_text(query_text, top_k=5):
    """
    Recherche de livres similaires basée sur la similarité cosine
    """
    # Générer l'embedding de la requête
    query_embedding = model.encode([query_text], convert_to_tensor=True).cpu().numpy()
    query_embedding = scaler.transform(query_embedding)
    
    # Calculer la similarité cosine avec tous les livres
    similarities = cosine_similarity(query_embedding, X_all)[0]
    
    # Trouver les top_k livres les plus similaires
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'title': dataset.iloc[idx]['Title'],
            'category': dataset.iloc[idx]['Category'],
            'similarity_score': similarities[idx],
            'description': dataset.iloc[idx]['Book_Description'][:200] if pd.notna(dataset.iloc[idx]['Book_Description']) else 'N/A'
        })
    
    return results

# 3. Fonction de prédiction de catégorie par similarité sémantique
def predict_category_semantic(query_text):
    """
    Prédit la catégorie d'un texte en utilisant la similarité cosine
    avec les profils de catégories
    """
    # Générer l'embedding de la requête
    query_embedding = model.encode([query_text], convert_to_tensor=True).cpu().numpy()
    query_embedding = scaler.transform(query_embedding)
    
    # Calculer la similarité avec chaque profil de catégorie
    similarities = {}
    for category, profile in category_profiles.items():
        similarity = cosine_similarity(query_embedding, profile.reshape(1, -1))[0][0]
        similarities[category] = similarity
    
    # Trier par similarité décroissante
    sorted_categories = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_categories

# 4. Clustering sémantique des livres
print("\n2. Clustering sémantique des livres...")
n_clusters = min(10, len(dataset['category_clean'].unique()))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_all)
dataset['semantic_cluster'] = cluster_labels

print(f"   → {n_clusters} clusters créés")
print("\nDistribution des clusters:")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"   Cluster {cluster_id}: {count} livres")

# 5. Exemples d'utilisation
print("\n" + "="*80)
print("EXEMPLES D'ANALYSE SÉMANTIQUE")
print("="*80)

# Exemple 1: Recherche sémantique
print("\n📚 Exemple 1: Recherche de livres similaires")
query_example = "artificial intelligence and machine learning"
print(f"Requête: '{query_example}'")
similar_books = find_similar_books_by_text(query_example, top_k=3)
print("\nLivres les plus similaires:")
for i, book in enumerate(similar_books, 1):
    print(f"\n{i}. {book['title']}")
    print(f"   Catégorie: {book['category']}")
    print(f"   Score de similarité: {book['similarity_score']:.4f}")
    print(f"   Description: {book['description']}...")

# Exemple 2: Prédiction de catégorie par similarité sémantique
print("\n\n🏷️  Exemple 2: Prédiction de catégorie par analyse sémantique")
query_example_2 = "detective mystery crime thriller investigation"
print(f"Requête: '{query_example_2}'")
category_predictions = predict_category_semantic(query_example_2)
print("\nTop 5 catégories les plus similaires:")
for i, (category, similarity) in enumerate(category_predictions[:5], 1):
    print(f"{i}. {category.title()}: {similarity:.4f}")

# Exemple 3: Analyse d'un cluster sémantique
print("\n\n🔍 Exemple 3: Analyse d'un cluster sémantique")
cluster_to_analyze = 0
cluster_books = dataset[dataset['semantic_cluster'] == cluster_to_analyze]
print(f"Cluster {cluster_to_analyze} ({len(cluster_books)} livres):")
print("\nCatégories principales dans ce cluster:")
category_distribution = cluster_books['category_clean'].value_counts().head(5)
for category, count in category_distribution.items():
    print(f"   - {category.title()}: {count} livres")
print("\nExemples de titres:")
for title in cluster_books['Title'].head(3):
    print(f"   • {title}")

# 6. Évaluation de la cohérence sémantique
print("\n\n📊 Évaluation de la cohérence sémantique par catégorie")
print("-" * 80)
for category in list(dataset['category_clean'].unique())[:5]:  # Top 5 catégories
    category_books = dataset[dataset['category_clean'] == category]
    if len(category_books) > 1:
        category_indices = np.where(dataset['category_clean'] == category)[0]
        category_embeddings = X_all[category_indices]
        # Calculer la similarité intra-catégorie moyenne
        intra_similarity = cosine_similarity(category_embeddings)
        # Moyenne des similarités (excluant la diagonale)
        mask = ~np.eye(intra_similarity.shape[0], dtype=bool)
        avg_similarity = intra_similarity[mask].mean()
        print(f"{category.title()[:30]:30} | Cohérence: {avg_similarity:.4f} | Livres: {len(category_books)}")

print("\n" + "="*80)
print("✅ Analyse sémantique contextuelle terminée!")
print("="*80)