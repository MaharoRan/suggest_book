import pandas as pd
import numpy as np

# Charger et nettoyer le dataset comme dans analysis.py
dataset = pd.read_csv('Book_Dataset_1.csv', sep=',', encoding='latin1')
dataset = dataset.drop_duplicates()

dataset = dataset[
    (dataset['Category'].str.strip() != '') &
    (dataset['Category'].str.lower() != 'default') &
    (dataset['Category'].str.lower() != 'add a comment')
]

def clean_text(text):
    if pd.isna(text):
        return ''
    return str(text).lower().strip()

dataset['category_clean'] = dataset['Category'].apply(clean_text)

print("="*80)
print("DIAGNOSTIC DU DATASET")
print("="*80)

print(f"\n📊 Total de livres: {len(dataset)}")
print(f"📚 Nombre de catégories uniques: {dataset['category_clean'].nunique()}")

print("\n📈 Distribution des catégories (top 20):")
print("-"*80)
category_counts = dataset['category_clean'].value_counts()
for i, (category, count) in enumerate(category_counts.head(20).items(), 1):
    percentage = (count / len(dataset)) * 100
    print(f"{i:2}. {category[:40]:40} | {count:4} livres ({percentage:5.2f}%)")

print(f"\n⚠️  Catégories avec < 5 livres: {sum(category_counts < 5)}")
print(f"⚠️  Catégories avec < 10 livres: {sum(category_counts < 10)}")
print(f"⚠️  Catégories avec < 20 livres: {sum(category_counts < 20)}")

print("\n📉 Statistiques de distribution:")
print("-"*80)
print(f"Moyenne de livres par catégorie: {category_counts.mean():.2f}")
print(f"Médiane: {category_counts.median():.0f}")
print(f"Écart-type: {category_counts.std():.2f}")
print(f"Min: {category_counts.min()}")
print(f"Max: {category_counts.max()}")

# Vérifier les catégories problématiques
print("\n🔍 Catégories avec le moins de livres:")
print("-"*80)
for category, count in category_counts.tail(10).items():
    print(f"   {category[:50]:50} | {count} livre(s)")

print("\n" + "="*80)
