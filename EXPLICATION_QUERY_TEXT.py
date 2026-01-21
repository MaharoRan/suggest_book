# ============================================================
# EXPLICATION DÉTAILLÉE : Construction de la requête
# ============================================================

"""
Question : Qu'est-ce que 'query_text' et comment est-il construit ?

Réponse : query_text est une PHRASE SÉMANTIQUE qui combine TOUTES 
les réponses du questionnaire en un texte cohérent.
"""

# ============================================================
# EXEMPLE CONCRET PAS À PAS
# ============================================================

# ÉTAPE 1 : L'utilisateur répond au questionnaire
# ------------------------------------------------

preferences = {
    # Questions ouvertes
    'description': "Je cherche un thriller psychologique avec suspense",
    'favorite_books': "Gone Girl, The Silent Patient",
    'avoid': "Romance",
    
    # Questions Likert (1-5)
    'intensity_action': 5,      # Très intense
    'intensity_romance': 1,      # Pas du tout
    'intensity_learning': 2,     # Un peu
    'complexity': 4              # Complexe
}


# ÉTAPE 2 : Construction de query_text
# ------------------------------------------------

def build_query_from_preferences(preferences):
    """
    Cette fonction transforme les réponses en UNE GRANDE PHRASE
    """
    parts = []
    
    # 1. Ajouter la description textuelle
    if preferences.get('description'):
        parts.append(preferences['description'])
        # Résultat : "Je cherche un thriller psychologique avec suspense"
    
    # 2. Ajouter les livres préférés
    if preferences.get('favorite_books'):
        parts.append(f"Livres similaires à : {preferences['favorite_books']}")
        # Résultat : "Livres similaires à : Gone Girl, The Silent Patient"
    
    # 3. Convertir les scores Likert en MOTS
    intensities = {
        'intensity_action': ['calme', 'paisible', 'modéré', 'intense', 'très intense'],
        'intensity_romance': ['sans romance', 'romance légère', 'romance présente', 
                             'romance importante', 'histoire d\'amour centrale'],
        'intensity_learning': ['divertissement pur', 'un peu éducatif', 'instructif', 
                              'très éducatif', 'essai pédagogique'],
        'complexity': ['très simple', 'accessible', 'standard', 'complexe', 'très complexe']
    }
    
    # Convertir chaque score en mot
    for key, descriptors in intensities.items():
        score = preferences.get(key, 3)  # Exemple : intensity_action = 5
        word = descriptors[score - 1]    # Index 4 (5-1) = "très intense"
        parts.append(word)
    
    # 4. Joindre tout avec des points
    query = ". ".join(parts)
    return query


# RÉSULTAT FINAL pour notre exemple :
query_text = """
Je cherche un thriller psychologique avec suspense. 
Livres similaires à : Gone Girl, The Silent Patient. 
très intense. 
sans romance. 
un peu éducatif. 
complexe.
"""

print("="*80)
print("QUERY_TEXT FINAL ENVOYÉ À SBERT :")
print("="*80)
print(query_text)
print()


# ============================================================
# ÉTAPE 3 : Ce query_text est utilisé dans 2 endroits
# ============================================================

print("="*80)
print("UTILISATION 1 : EMBEDDINGS SBERT (Recommandation)")
print("="*80)

print("""
1. Le système encode query_text en vecteur 384D
   query_emb = model.encode(query_text)
   
2. Compare ce vecteur avec les 781 livres du dataset
   scores = cosine_similarity(query_emb, tous_les_livres_embeddings)
   
3. Trouve les 3 livres les plus similaires
   top_3 = livres avec scores les plus élevés
""")


print("="*80)
print("UTILISATION 2 : PROMPT GENAI (Synthèse)")
print("="*80)

print("""
Le query_text est AUSSI envoyé à Gemini pour générer la synthèse.

PROMPT ENVOYÉ À GEMINI :
┌────────────────────────────────────────────────────────┐
│ Tu es un conseiller littéraire.                        │
│                                                        │
│ PROFIL : Je cherche un thriller psychologique avec    │
│          suspense. Livres similaires à : Gone Girl,   │
│          The Silent Patient. très intense. sans       │
│          romance. un peu éducatif. complexe.          │
│                                                        │
│ LIVRE : The Girl on the Train (Mystery Thriller)      │
│         A psychological thriller about...             │
│                                                        │
│ TÂCHE (120 mots max) :                               │
│ 1. Pourquoi ce livre correspond (3 phrases)          │
│ 2. 2 aspects clés couverts                           │
│ 3. Suggestion de 2 livres similaires                 │
└────────────────────────────────────────────────────────┘

Gemini analyse le PROFIL (query_text) et le LIVRE recommandé,
puis génère une synthèse personnalisée.
""")


# ============================================================
# EXEMPLE COMPLET DE FLUX
# ============================================================

print("="*80)
print("FLUX COMPLET - EXEMPLE RÉEL")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────┐
│ 1. QUESTIONNAIRE UTILISATEUR                                 │
└─────────────────────────────────────────────────────────────┘
   Input : "thriller psychologique"
           "Gone Girl"
           Scores : [5, 1, 2, 4]
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. CONSTRUCTION QUERY_TEXT                                   │
└─────────────────────────────────────────────────────────────┘
   query_text = "Je cherche un thriller psychologique avec 
                 suspense. Livres similaires à : Gone Girl, 
                 The Silent Patient. très intense. sans 
                 romance. un peu éducatif. complexe."
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3A. SBERT ENCODING                                          │
└─────────────────────────────────────────────────────────────┘
   query_emb = [0.24, -0.15, 0.89, ..., 0.33]  (384 nombres)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. SIMILARITÉ COSINUS                                        │
└─────────────────────────────────────────────────────────────┘
   Livre 1 : The Girl on the Train      → Score : 0.8634
   Livre 2 : Sharp Objects               → Score : 0.8421
   Livre 3 : Before I Go to Sleep        → Score : 0.8189
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. GÉNÉRATION SYNTHÈSE GENAI                                │
└─────────────────────────────────────────────────────────────┘
   Prompt à Gemini :
   ┌─────────────────────────────────────────────────────────┐
   │ PROFIL : Je cherche un thriller psychologique...       │
   │ LIVRE : The Girl on the Train                          │
   │ TÂCHE : Explique pourquoi ce livre correspond...       │
   └─────────────────────────────────────────────────────────┘
                            ↓
   Gemini génère :
   "The Girl on the Train correspond parfaitement à votre 
    recherche de thriller psychologique intense. Le récit 
    utilise une narration non-fiable qui crée un suspense 
    constant, similaire à Gone Girl que vous avez apprécié.
    
    Aspects couverts :
    ✓ Suspense psychologique maximal (intensité 5/5)
    ✓ Absence totale de romance (score 1/5 respecté)
    
    Recommandations similaires :
    - The Woman in the Window (A.J. Finn)
    - The Last Mrs. Parrish (Liv Constantine)"
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. AFFICHAGE À L'UTILISATEUR                                │
└─────────────────────────────────────────────────────────────┘
   [SYNTHESE PERSONNALISEE - GenAI]
   The Girl on the Train correspond parfaitement...
   
   [TOP 3 RECOMMANDATIONS]
   1. The Girl on the Train | Score: 86.34%
   2. Sharp Objects          | Score: 84.21%
   3. Before I Go to Sleep   | Score: 81.89%
""")


# ============================================================
# POINTS CLÉS À RETENIR
# ============================================================

print("\n" + "="*80)
print("POINTS CLÉS")
print("="*80)

print("""
1. query_text = TOUTES les réponses du questionnaire en UNE phrase

2. Scores Likert CONVERTIS en mots :
   5 (action) → "très intense"
   1 (romance) → "sans romance"
   
3. query_text utilisé pour :
   a) SBERT : Trouver les livres similaires (matching sémantique)
   b) GENAI : Expliquer POURQUOI les livres correspondent
   
4. UN SEUL appel API Gemini avec :
   - Le profil (query_text)
   - Le livre recommandé (titre + description)
   - Une tâche claire (120 mots max)
   
5. Résultat : Recommandations + Synthèse personnalisée
""")


print("\n" + "="*80)
print("EXEMPLE DE TRANSFORMATION LIKERT → MOTS")
print("="*80)

likert_examples = [
    ("intensity_action", 5, "très intense"),
    ("intensity_action", 3, "modéré"),
    ("intensity_action", 1, "calme"),
    ("intensity_romance", 5, "histoire d'amour centrale"),
    ("intensity_romance", 1, "sans romance"),
    ("complexity", 4, "complexe"),
    ("complexity", 2, "accessible"),
]

for key, score, word in likert_examples:
    print(f"  {key:20} | Score: {score}/5 → '{word}'")


print("\n" + "="*80)
print("RÉSUMÉ EN 3 PHRASES")
print("="*80)

print("""
1. query_text = Fusion de TOUTES vos réponses en une phrase cohérente

2. SBERT utilise query_text pour trouver les livres les + similaires

3. Gemini utilise query_text + livre recommandé pour expliquer le match
""")
