import pickle
import re
import os

# Chemins des modèles
model_path = "../models/sentiment_model.pkl"
vectorizer_path = "../models/tfidf_vectorizer.pkl"

# Vérification de l'existence des fichiers
if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
    print("❌ Erreur: Fichier modèle ou vectoriseur introuvable. Vérifiez les chemins.")
    exit(1)

# Chargement des modèles
with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Nettoyage de texte
def clean_tweet(text):
    text = re.sub(r'@\w+', '', text)  # Mentions
    text = re.sub(r'http\S+', '', text)  # URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Caractères spéciaux
    return text.lower().strip()

# Prédiction de sentiment
def predict_sentiment(text):
    if not hasattr(vectorizer, "idf_"):
        print("❌ Erreur: Problème de configuration du vectoriseur.")
        return "Erreur"
    
    cleaned = clean_tweet(text)
    transformed = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(transformed)
    return "😊 Positif" if prediction[0] == 1 else "😞 Négatif"

# Interface en ligne de commande
print("\n=== Analyseur de Sentiment pour Tweets ===")
print("Tapez 'q' pour quitter\n")

while True:
    tweet = input("Entrez votre tweet : ")
    
    if tweet.lower() == 'q':
        print("\nMerci d'avoir utilisé l'analyseur ! 👋")
        break
        
    if not tweet.strip():
        print("⚠️ Veuillez entrer un texte valide")
        continue
        
    resultat = predict_sentiment(tweet)
    if resultat != "Erreur":
        print(f"\nRésultat : {resultat}\n")
