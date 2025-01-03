import pandas as pd
import numpy as np
from sklearn import datasets
import streamlit as st
from Model import Model
from LinearSVCModel import LinearSVCModel
from SGDclassifierModel import SGDclassifierModel
from Data import Data
import shap
import matplotlib.pyplot as plt


# Ici on gère l'import du CSV
st.title("Machine Learning Model Explanation Project")
st.write('''
To begin, import your data in csv form. The first line of the csv are the attributes and the last column is the target
''')
upload = st.file_uploader("Choose your data", type="csv")

if upload is not None:
    data = Data(upload)

    attributes = data.collectDataAttributes() # On extrait les noms différents attributs du JDD (ex : *taille* de x, *longueur* de x etc)
    classes = data.collectDataClasses() # On extrait les différentes classes du JDD (ex : chat, chien, vache)

    st.write('''
    Welcome to an example of a prediction application with a Machine Learning model.
    What you're seeing is an example of the interface which will later be further enhanced with several elements of explainability.
    For now, only the modularity of the data set is in place although its selection itself isn't yet.
    ''')

    # On définie de manière modulaire nos sliders de customisation des paramètres d'attributs à gauche dans l'interface
    st.sidebar.header("Slide to Modify Attribute Values")
    user_inputs = {} # Dictionnaire pour stocker les valeurs définies par l'utilisateur via les sliders pour chaque attribut
    for attribute in attributes: # Boucle qui va créer un slider de customisation pour chaque attribut précédemment extraits présents dans le jeu de donnée
        min_val, max_val = data.collectMinMaxValues(attribute)
        user_inputs[attribute] = st.sidebar.slider( 
            f"{attribute}", # Pour afficher le nom de la colonne/le nom de l'attribut correspondant au dessus du slider créé
            min_value=float(min_val), # Défini le min du slider selon la plus petite valeur contenue dans la colonne de l'attribut
            max_value=float(max_val), # Défini le max du slider selon la plus grande valeur contenue dans la colonne de l'attribut
            value=float((min_val + max_val) / 2),  # Pour placer le slider au milieu par défaut /Pour définir par défaut la valeur à la moitié du champ possible
        )

    st.subheader("Attribute Values Being Subject to the Prediction")
    st.write("(Slide to consult undisplayed attributes if needed)")
    st.table([user_inputs]) # On affiche le tableau qui contient tout les attributs et toutes les valeurs associés qu'on a définié via les sliders

    st.subheader("Potential Prediction Results: ")
    st.table([{"Class": cls} for cls in classes]) # On affiche dans un tableau toutes les Class potentielles à la prédiction qui existent dans le fichier csv

    user_inputs_df = pd.DataFrame([user_inputs]) # On converti les entrées utilisateurs(valeurs des sliders) en DataFrame pour la prédiction

    # On crée un dictionnaire avec la liste des modèles en place
    modelOptions = {
        "LinearSVC": LinearSVCModel,
        "SGDclassifier": SGDclassifierModel,
    }

    #On crée une boîte de sélection dans l'interface pour rendre le choix du modèle flexible 
    #(Permet aussi de changer de modèle sans avoir à redémarrer l'appli, bien plus pratique à l'utilisation)
    modelUserSelection = st.selectbox("Select a Model (LinearSVC is used by default)", list(modelOptions.keys()))
    
    #On met en place le modèle choisi et sa prédiction puis l'effectue
    chosenModel = modelOptions[modelUserSelection]
    model = Model(chosenModel(), data, user_inputs_df)
    prediction = model.predict()
    concreteModel = model.getConcreteModel()


    st.write("The predicted class is :",prediction) # On affiche la prédiction

    #crée les shap values correspondant au model
    X= data.dfX
    explainer = shap.Explainer(concreteModel, X)
    shap_values = explainer(user_inputs_df)

    #si classes du dataset ne sont pas bianire (ex:van, saab, opel, bus)
    if classes.__len__() > 2:
        #on recherche l'indice correspondant à la prédiction pour donner la bonne dimension de shap values
        for i in range(classes.__len__()):
            if classes[i] == prediction:
                raw_prediction = i
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0][:,raw_prediction], show=False, max_display=50)
        st.pyplot(fig)
    #sinon
    else:

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False, max_display=50)
        st.pyplot(fig)