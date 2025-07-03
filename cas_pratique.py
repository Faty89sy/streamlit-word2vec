import streamlit as st
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Configuration et titre personnalisé
st.set_page_config(page_title="Word2Vec Interactive", layout="centered")
st.markdown("<h1 style='text-align: center; color: darkblue;'>Application Word2Vec</h1>", unsafe_allow_html=True)
st.markdown("---")

# Charger le tokenizer sauvegardé depuis tokenizer.json
with open("tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())

# Définir vocab_size, word2idx et idx2word
vocab_size = tokenizer.num_words if tokenizer.num_words is not None else len(tokenizer.word_index) + 1
word2idx = tokenizer.word_index
idx2word = {v: k for k, v in word2idx.items()}

st.title("Modèle Word2Vec")

from tensorflow.keras.models import load_model
model = load_model("word2vec.h5")

vectors = model.layers[0].trainable_weights[0].numpy()
import numpy as np
from sklearn.preprocessing import Normalizer

def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

# Interface Streamlit pour trouver les mots les plus proches
st.subheader("Trouver les mots les plus proches")

mot_saisi = st.text_input("Entrez un mot (en anglais) pour voir les plus proches :", "zombie")
nb_mots = st.slider("Nombre de mots similaires à afficher :", 1, 20, 10)

if mot_saisi in word2idx:
    resultats = find_closest(word2idx[mot_saisi], vectors, nb_mots)
    st.write(f"Mots les plus proches de **{mot_saisi}** :")
    for index_word in resultats:
        st.write(f"- {idx2word[index_word[1]]} (similarité : {round(index_word[0], 3)})")
else:
    st.error("Le mot n'existe pas dans le vocabulaire. Essayez un autre mot.")

# Interface Streamlit pour les analogies de type "roi - homme + femme ≈ reine"
st.subheader("Analogies sémantiques (A - B + C)")

col1, col2, col3 = st.columns(3)
with col1:
    mot1 = st.text_input("Mot A", "king")
with col2:
    mot2 = st.text_input("Mot B", "man")
with col3:
    mot3 = st.text_input("Mot C", "woman")

nb_resultats = st.slider("Nombre de suggestions", 1, 10, 5)

if all(m in word2idx for m in [mot1, mot2, mot3]):
    resultats = compare(word2idx[mot1], word2idx[mot2], word2idx[mot3], vectors, nb_resultats)
    st.write(f"Résultats pour : **{mot1} - {mot2} + {mot3}**")
    for index_word in resultats:
        st.write(f"- {idx2word[index_word[1]]} (similarité : {round(index_word[0], 3)})")
else:
    st.warning("Un ou plusieurs mots ne sont pas dans le vocabulaire.")

# Interface Streamlit pour les analogies sémantiques
st.subheader("Comparer des mots (analogie sémantique)")

word1 = st.text_input("Mot 1 (positif)", "king")
word2 = st.text_input("Mot 2 (négatif)", "man")
word3 = st.text_input("Mot 3 (positif)", "woman")
nb_results = st.slider("Nombre de mots similaires à afficher :", 1, 20, 5, key="analogie")

if all(w in word2idx for w in [word1, word2, word3]):
    resultats = compare(word2idx[word1], word2idx[word2], word2idx[word3], vectors, nb_results)
    st.write(f"Résultat de l'analogie **{word1} - {word2} + {word3}** :")
    for index_word in resultats:
        st.write(f"- {idx2word[index_word[1]]} (similarité : {round(index_word[0], 3)})")
else:
    st.warning("Un ou plusieurs mots sont absents du vocabulaire.")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>Projet Deep Learning – Formation Data Scientist </p>", unsafe_allow_html=True)