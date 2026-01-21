# 📚 GUIDE COMPLET - Système de Recommandation Littéraire

## EFREI M1 - Data Engineering & IA Générative

---

## 🎯 VUE D'ENSEMBLE

Ce système répond aux **Exigences Fonctionnelles (EF)** du projet en combinant :

- **NLP local** (SBERT - coût zéro)
- **Similarité cosinus** (matching sémantique)
- **GenAI stratégique** (enrichissement conditionnel + synthèse finale)

---

## 📋 ARCHITECTURE DU SYSTÈME

```
┌─────────────────────────────────────────────────────────────┐
│  EF1 : ACQUISITION DONNÉES (Questionnaire)                  │
│  ├─ Questions ouvertes (description, livres préférés)       │
│  └─ Questions Likert 1-5 (action, romance, complexité)      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  EF2 : MOTEUR NLP SÉMANTIQUE (Local, Gratuit)              │
│  ├─ Référentiel : 700+ livres (EF2.1)                      │
│  ├─ SBERT Embeddings : 384 dimensions (EF2.2)              │
│  └─ Similarité Cosinus (EF2.3)                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  EF4.1 : AUGMENTATION PRE-PROCESSING (Conditionnelle)      │
│  └─ Si texte < 5 mots → GenAI enrichit le contexte         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  EF3 : SCORING & RECOMMANDATION                             │
│  ├─ Score pondéré : 80% similarité + 20% intensité (EF3.1) │
│  └─ Top 3 livres recommandés (EF3.2)                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  EF4.2-4.3 : SYNTHÈSE GENAI (1 seul appel)                 │
│  ├─ Explication personnalisée (EF4.2)                      │
│  └─ Executive Summary (EF4.3)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 EXPLICATIONS DÉTAILLÉES DES MODULES

### **MODULE 1 : EF1 - Acquisition de la Donnée**

#### **Fonction : `collect_user_preferences()`**

**Ce qu'elle fait :**

- Pose un questionnaire **hybride** à l'utilisateur
- Combine questions **ouvertes** et échelles **Likert (1-5)**

**Questions posées :**

1. **Questions ouvertes** (EF1.1)

   ```python
   - Description du livre recherché (libre)
   - Livres préférés (références)
   - Genres à éviter (exclusions)
   ```

2. **Questions numériques** (Échelle Likert 1-5)
   ```python
   - Intensité d'action souhaitée
   - Intérêt pour la romance
   - Importance de l'apprentissage
   - Complexité narrative
   ```

**Pourquoi c'est important :**

- Les réponses ouvertes capturent le **contexte sémantique**
- Les scores Likert permettent une **pondération quantitative**
- Combinaison = analyse riche et nuancée

**Stockage (EF1.2) :**

```json
{
  "description": "Je cherche un thriller psychologique...",
  "favorite_books": "Gone Girl, The Silent Patient",
  "intensity_action": 5,
  "intensity_romance": 2,
  "timestamp": "2025-12-10T14:30:00"
}
```

---

### **MODULE 2 : EF2 - Moteur NLP Sémantique**

#### **Fonction : `load_knowledge_base()`** (EF2.1)

**Ce qu'elle fait :**

- Charge le dataset de livres (CSV)
- Nettoie les données (supprime doublons, catégories invalides)
- Crée un **référentiel de connaissances** structuré

**Transformation des données :**

```python
# Avant
Title: "The Great Gatsby"
Category: "Fiction"
Description: "A story of wealth and love..."

# Après nettoyage
text_full: "the great gatsby. fiction. a story of wealth and love..."
```

**Pourquoi `text_full` ?**

- Combine **titre + genre + description** en UN texte
- Permet à SBERT de capturer le **contexte complet** du livre
- Plus d'information = meilleur embedding

---

#### **Fonction : `load_sbert_and_embeddings()`** (EF2.2)

**Ce qu'elle fait :**

- Charge le modèle **SentenceTransformer** (SBERT)
- Génère des **embeddings** (vecteurs 384D) pour chaque livre
- Cache les résultats pour éviter de recalculer

**Qu'est-ce qu'un embedding ?**

```
Texte : "the great gatsby. fiction. a story of wealth..."
         ↓ SBERT
Vecteur : [0.24, -0.15, 0.89, ..., 0.33]  (384 nombres)
```

**Pourquoi SBERT ?**

- **Local** : Aucun coût, aucune API externe
- **Rapide** : Traite 700 livres en ~30 secondes
- **Sémantique** : Capture le **sens** du texte, pas juste les mots

**Mise en cache :**

```python
# Première exécution : génère et sauvegarde
embeddings_books.pkl  # 700 livres × 384D ≈ 2 MB

# Exécutions suivantes : charge depuis le fichier
# Gain de temps : 30s → 1s
```

---

#### **Fonction : `calculate_weighted_similarity()`** (EF2.3 + EF3.1)

**Ce qu'elle fait :**

1. Calcule la **similarité cosinus** entre la requête et un livre (EF2.3)
2. Applique une **pondération** basée sur les scores Likert (EF3.1)

**Formule mathématique :**

```
Similarité Cosinus = (A · B) / (||A|| × ||B||)

Où :
- A = vecteur embedding de la requête utilisateur
- B = vecteur embedding du livre
- · = produit scalaire
- ||X|| = norme du vecteur

Résultat : Nombre entre 0 (aucune similarité) et 1 (identique)
```

**Exemple concret :**

```python
Requête : "thriller psychologique avec suspense"
Livre 1 : "Gone Girl" (thriller psychologique)
Livre 2 : "Pride and Prejudice" (romance classique)

Similarités cosinus :
- Gone Girl        : 0.87  (très similaire)
- Pride & Prejudice : 0.23  (peu similaire)
```

**Pondération par intensités (EF3.1) :**

```python
# Moyenne des scores Likert
avg_intensity = (5 + 2 + 3 + 4) / 4 / 5 = 0.70  (normalisé 0-1)

# Score final pondéré
weighted_score = 0.8 × cosine_sim + 0.2 × avg_intensity

# Exemple
Gone Girl : 0.8 × 0.87 + 0.2 × 0.70 = 0.836
```

**Pourquoi cette pondération ?**

- **80% sémantique** : Le sens du texte est prioritaire
- **20% intensité** : Les préférences numériques affinent le résultat
- Balance entre matching contextuel et préférences quantitatives

---

### **MODULE 3 : EF4.1 - Augmentation Pre-Processing**

#### **Fonction : `enrich_short_query()`**

**Ce qu'elle fait :**

- Détecte si la requête utilisateur est **trop courte** (< 5 mots)
- Si OUI → Appelle GenAI pour enrichir le contexte
- Si NON → Utilise le texte original

**Exemple d'enrichissement :**

```python
# Texte utilisateur (3 mots)
Input : "thriller suspense"

# Après enrichissement GenAI
Output : "thriller suspense avec intrigue psychologique complexe,
          retournements de situation et ambiance sombre et oppressante"
```

**Pourquoi c'est utile ?**

- SBERT fonctionne mieux avec du **contexte riche**
- 3 mots = embedding pauvre, résultats moins précis
- Enrichissement = plus d'informations pour le matching

**Usage conditionnel (EF4.1) :**

```python
if len(texte.split()) < 5:
    # UN SEUL appel API si nécessaire
    texte_enrichi = appel_genai(texte)
else:
    # Aucun coût si texte suffisant
    texte_enrichi = texte
```

**Fallback sans GenAI :**

- Si pas de clé API → enrichissement basique local
- Garantit que le système fonctionne **toujours**

---

### **MODULE 4 : EF3.2 - Recommandation Top 3**

#### **Fonction : `recommend_books()`**

**Ce qu'elle fait :**

1. Construit une requête sémantique complète
2. Encode la requête en embedding
3. Calcule les scores pour TOUS les livres
4. Retourne les **3 meilleurs**

**Pipeline complet :**

```python
# 1. Construction de la requête
preferences = {
    'description': "thriller psychologique",
    'favorite_books': "Gone Girl",
    'intensity_action': 5
}
         ↓
query_text = "thriller psychologique. Livres similaires à Gone Girl.
              très intense. romance légère. instructif. complexe"

# 2. Enrichissement (si < 5 mots)
query_enriched = enrich_short_query(query_text)

# 3. Embedding
query_emb = SBERT.encode(query_enriched)  # [0.21, -0.45, ..., 0.89]

# 4. Calcul des scores
scores = []
for livre in tous_les_livres:
    score = calculate_weighted_similarity(query_emb, livre.embedding)
    scores.append(score)

# 5. Top 3
indices_top3 = argsort(scores)[:3]
```

**Format des résultats :**

```python
[
    {
        'rank': 1,
        'title': 'The Silent Patient',
        'genre': 'Mystery Thriller',
        'similarity_score': 0.8634,
        'description': '...'
    },
    {
        'rank': 2,
        'title': 'Sharp Objects',
        'genre': 'Mystery Thriller',
        'similarity_score': 0.8421,
        'description': '...'
    },
    {
        'rank': 3,
        'title': 'The Girl on the Train',
        'genre': 'Mystery Thriller',
        'similarity_score': 0.8189,
        'description': '...'
    }
]
```

---

### **MODULE 5 : EF4.2 & EF4.3 - Synthèse GenAI**

#### **Fonction : `generate_personalized_summary()`**

**Ce qu'elle fait :**

- Génère une **synthèse personnalisée** expliquant les recommandations
- **UN SEUL appel API** (économie de coûts)
- Format **Executive Summary** professionnel

**Prompt envoyé à Gemini :**

```
Tu es un conseiller littéraire expert.

PROFIL LECTEUR :
thriller psychologique avec suspense intense...

LIVRE RECOMMANDÉ (Score : 0.863) :
- Titre : The Silent Patient
- Genre : Mystery Thriller
- Résumé : A shocking psychological thriller...

MISSION :
1. Explique POURQUOI ce livre correspond (4-5 phrases)
2. Identifie 2-3 aspects parfaitement couverts
3. Propose une orientation de lecture complémentaire
```

**Exemple de synthèse générée :**

```
🧠 SYNTHÈSE PERSONNALISÉE :

"The Silent Patient" correspond parfaitement à votre profil de lecteur
recherchant une tension psychologique intense. Le récit combine l'intrigue
criminelle que vous appréciez dans "Gone Girl" avec une profondeur
psychologique encore plus marquée, explorant les traumatismes et le silence
comme armes narratives.

ASPECTS COUVERTS :
✓ Suspense psychologique très intense (correspond à votre score 5/5)
✓ Complexité narrative avec retournements surprenants
✓ Ambiance sombre et oppressante

ORIENTATION COMPLÉMENTAIRE :
Pour enrichir votre parcours, explorez ensuite les thrillers nordiques
(Stieg Larsson, Jo Nesbø) qui prolongent cette atmosphère tout en ajoutant
une dimension sociale et politique.
```

**Pourquoi UN SEUL appel ?**

- Coût maîtrisé : ~$0.001 par requête
- Conformité EF4.2 : "Un seul appel API pour la sortie finale"
- Valeur maximale : synthèse complète en une fois

---

## 🔧 CONCEPTS TECHNIQUES À RETENIR

### **1. Embeddings (Vecteurs Sémantiques)**

**Définition :**
Transformation d'un texte en vecteur de nombres qui capture son **sens**.

```python
Texte → [n1, n2, n3, ..., n384]
```

**Propriété magique :**
Des textes **similaires** ont des vecteurs **proches** dans l'espace.

**Exemple :**

```
"chat"          → [0.8, 0.2, -0.1, ...]
"chien"         → [0.7, 0.3, -0.2, ...]  (proche de chat)
"ordinateur"    → [-0.1, -0.5, 0.9, ...] (loin de chat)
```

---

### **2. Similarité Cosinus**

**Définition :**
Mesure l'**angle** entre deux vecteurs (0° = identique, 90° = orthogonal).

**Formule simplifiée :**

```
cos(θ) = Somme(A × B) / (Longueur(A) × Longueur(B))
```

**Interprétation :**

```
1.0  = Textes identiques
0.8+ = Très similaires (recommandation forte)
0.5  = Moyennement similaires
0.0  = Aucun lien
```

---

### **3. Différence SBERT vs Word2Vec/GloVe**

| Aspect      | Word2Vec/GloVe   | SBERT               |
| ----------- | ---------------- | ------------------- |
| **Niveau**  | Mots individuels | Phrases/Paragraphes |
| **Context** | Fenêtre locale   | Contexte global     |
| **Taille**  | ~300D            | 384-768D            |
| **Usage**   | Analyse de mots  | Matching de textes  |

**Exemple :**

```python
# Word2Vec
"apple" → vecteur fruit/tech (ambiguïté)

# SBERT
"I love eating apples" → vecteur clairement fruit
"Apple released iPhone" → vecteur clairement tech
```

---

### **4. Pondération des Scores**

**Pourquoi 80/20 ?**

```python
score_final = 0.8 × similarité_cosinus + 0.2 × intensité_likert
```

**Justification :**

- **80% sémantique** : Le texte contient l'information principale
- **20% intensité** : Ajuste selon les préférences quantifiées
- Évite que des scores Likert "forts" masquent une faible similarité textuelle

**Exemple comparatif :**

```
Livre A : Similarité 0.9, Intensité 0.5
Score = 0.8×0.9 + 0.2×0.5 = 0.82

Livre B : Similarité 0.5, Intensité 0.9
Score = 0.8×0.5 + 0.2×0.9 = 0.58

→ Livre A gagne (sémantique prime)
```

---

## 🚀 UTILISATION DU SYSTÈME

### **Sans GenAI (Gratuit, Local)**

```bash
python book_recommendation_system.py
```

Le système fonctionne **entièrement** avec SBERT local.

### **Avec GenAI (Enrichissement + Synthèse)**

```bash
# Windows
set GEMINI_API_KEY=votre_clé_ici
python book_recommendation_system.py

# Linux/Mac
export GEMINI_API_KEY=votre_clé_ici
python book_recommendation_system.py
```

**Obtenir une clé Gemini (gratuite) :**

1. Aller sur https://makersuite.google.com/app/apikey
2. Créer un projet
3. Générer une clé API
4. Limite gratuite : 60 requêtes/minute

---

## 📊 FLUX DE DONNÉES COMPLET

```
UTILISATEUR
    ↓ Répond au questionnaire
┌─────────────────────────────────┐
│ Questions ouvertes + Likert     │
│ → JSON structuré                │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Construction requête sémantique │
│ → "thriller + intense + ..."    │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Enrichissement (si texte court) │
│ → GenAI ajoute contexte         │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Embedding SBERT                 │
│ → Vecteur 384D                  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Calcul similarité × 700 livres  │
│ → Scores pondérés               │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Top 3 recommandations           │
│ → Livres classés par score      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Synthèse GenAI (optionnel)      │
│ → Explication personnalisée     │
└─────────────────────────────────┘
    ↓
RÉSULTATS AFFICHÉS
```

---

## 🎓 POINTS CLÉS À RETENIR POUR L'EXAMEN

### **1. Architecture Hybride**

- **NLP local** (SBERT) = Base gratuite et performante
- **GenAI stratégique** = Enrichissement ciblé et synthèse finale
- **Équilibre coût/valeur** : 95% local, 5% GenAI

### **2. Conformité aux Exigences**

- **EF1** ✅ : Questionnaire hybride (ouvert + Likert)
- **EF2** ✅ : Référentiel + SBERT + Cosinus
- **EF3** ✅ : Scoring pondéré + Top 3
- **EF4** ✅ : 2 appels GenAI max (enrichissement + synthèse)

### **3. Avantages de l'Approche**

- **Zéro coût en mode local**
- **Scalable** : Peut traiter 10K+ livres
- **Personnalisé** : Chaque utilisateur a des résultats uniques
- **Explicable** : Scores de similarité transparents

### **4. Améliorations Possibles**

- **Filtrage collaboratif** : Ajouter les préférences d'autres utilisateurs
- **Fine-tuning SBERT** : Entraîner sur des données littéraires
- **Interface Web** : Flask/Streamlit pour l'IHM
- **Visualisation** : Graphiques radar des scores

---

## 📝 FICHIERS GÉNÉRÉS

```
book_recommendation_system.py        # Code principal
user_preferences.json                # Réponses utilisateur
embeddings_books.pkl                 # Cache des embeddings
recommendation_results.json          # Résultats finaux
```

---

## ✅ CHECKLIST DE COMPRÉHENSION

- [ ] Je comprends ce qu'est un **embedding**
- [ ] Je peux expliquer la **similarité cosinus**
- [ ] Je sais pourquoi SBERT > Word2Vec pour ce cas
- [ ] Je comprends la **pondération 80/20**
- [ ] Je peux justifier l'**usage conditionnel** de GenAI
- [ ] Je sais mapper chaque fonction à une exigence (EF1-EF4)

---

**Bon courage pour ton projet ! 🚀**
