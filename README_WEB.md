# 🌐 Interface Web - Système de Recommandation Littéraire

## 📋 Installation

1. **Installer Flask** (si pas déjà installé) :

```bash
pip install flask
```

Ou installer toutes les dépendances :

```bash
pip install -r requirements.txt
```

## 🚀 Lancement

1. **Démarrer le serveur** :

```bash
python app.py
```

2. **Ouvrir dans le navigateur** :

```
http://localhost:5000
```

## 📱 Utilisation

1. Remplissez le questionnaire :

   - 3 questions ouvertes (texte libre)
   - 4 échelles Likert (1-5)

2. Cliquez sur "Obtenir mes recommandations"

3. Résultats affichés :
   - ✅ Synthèse personnalisée (GenAI)
   - ✅ Top 3 recommandations avec scores
   - ✅ Détails de chaque livre

## 🎨 Fonctionnalités

- ✅ Interface moderne et responsive
- ✅ Formulaire interactif avec échelles Likert visuelles
- ✅ Chargement asynchrone (pas de rechargement de page)
- ✅ Affichage élégant des résultats
- ✅ Gestion des erreurs
- ✅ Une seule page (pas de menu)

## 🛠️ Architecture

```
app.py              → Serveur Flask (backend)
templates/
  └── index.html    → Interface web (frontend)
requirements.txt    → Dépendances Python
```

## 🔧 Configuration

La clé API Gemini est codée en dur dans `app.py` :

```python
api_key = "votre_clé"
```

Pour utiliser une variable d'environnement :

```bash
set GEMINI_API_KEY=votre_clé
```

Puis modifier `app.py` :

```python
api_key = os.getenv("GEMINI_API_KEY")
```

## 📊 Exemple de flux

1. Utilisateur remplit le formulaire
2. Clic sur "Obtenir mes recommandations"
3. Affichage du loading
4. Appel AJAX vers `/recommend`
5. Backend :
   - Construit query_text
   - Calcule similarités SBERT
   - Génère synthèse GenAI
6. Affichage des résultats dans la même page

## 🎯 Points clés

- **Aucun rechargement** : AJAX pour une expérience fluide
- **Design moderne** : Gradient violet, cartes interactives
- **Échelles visuelles** : Boutons Likert cliquables
- **Responsive** : Fonctionne sur mobile/tablette/desktop
- **Simple** : Une seule page, pas de navigation complexe
