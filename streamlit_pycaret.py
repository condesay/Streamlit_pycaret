import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.nlp import *

def main():
    st.title('Classification avec Pycaret')

    # Charger les données
    @st.cache(allow_output_mutation=True)
    def load_data(file):
        data = pd.read_csv(file)
        return data

    file = st.file_uploader("Upload file", type=["csv"])
    if file is not None:
        df = load_data(file)

        # Afficher les premières lignes du fichier
        st.write("## Les premières lignes du fichier:")
        st.write(df.head())

        # Afficher des informations sur les colonnes
        st.write("## Informations sur les colonnes:")
        st.write(df.info())

        # Afficher les statistiques descriptives
        st.write("## Statistiques descriptives:")
        st.write(df.describe())

        # Afficher la taille des données
        st.write("## La taille des données:")
        st.write(df.shape)

        # Sélectionner les features et la target
        def select_features(df):
            features = st.multiselect("Sélectionnez les features", df.columns.tolist())
            return features

        def select_target(df):
            target = st.selectbox("Sélectionnez la target", df.columns.tolist(),key="unique_key")
            return target

        st.write("## Choix de la target et des features:")
        target = select_target(df)
        features = select_features(df)

        # Prétraiter les données
        def preprocess_data(df):
            df = df.sample(1000, random_state=786).reset_index(drop=True)
            return df

        df = preprocess_data(df)
        st.write("## Prétraitement:")
        st.write(df)

        # Configurer l'expérience PyCaret
        def setup_experiment(df):
            exp_nlp101 = setup(data=df, target=target, session_id=123)
            return exp_nlp101

        exp_nlp101 = setup_experiment(df)

        # Créer le modèle LDA
        @st.cache(allow_output_mutation=True)
        def create_lda_model():
            ldafr = create_model('lda')
            return ldafr

        ldafr = create_lda_model()

        # Assigner les topics aux documents
        lda_results = assign_model(ldafr)

        # Évaluer le modèle
        evaluate_model(ldafr)

        # Ajuster le modèle
        @st.cache(allow_output_mutation=True)
        def tune_lda_model():
            tuned_classification = tune_model(model='lda', multi_core=True, supervised_target=target)
            return tuned_classification

        tuned_classification = tune_lda_model()

        # Visualiser les résultats
        st.title("Topic Modeling avec PyCaret et Streamlit")
        st.subheader("Word Cloud")
        stop_words = stopwords.words('french')

        tx1=""
        for info in df[target]:
            tx1 = tx1 + str(info) + " "

        wc = WordCloud(
            background_color='white',
            max_words=2000,
            stopwords=stop_words
        )

        wc.generate(str(tx1))

        plt.imshow(wc)
        plt.axis('off')
        st.pyplot()

if __name__ == '__main__':
    main()
