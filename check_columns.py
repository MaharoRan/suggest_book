import pandas as pd

# Tester différents encodages
encodings = ['latin1', 'utf-8', 'cp1252', 'iso-8859-1']

for encoding in encodings:
    try:
        df = pd.read_csv('Book_Dataset_1.csv', sep=',', encoding=encoding, nrows=5)
        print(f"\n✅ Encodage '{encoding}' fonctionne!")
        print(f"Colonnes: {list(df.columns)}")
        print(f"\nPremières lignes:")
        print(df.head(2))
        break
    except Exception as e:
        print(f"❌ Encodage '{encoding}' échoué: {e}")
