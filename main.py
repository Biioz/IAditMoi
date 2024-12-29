import pandas as pd #permet de gérer, trier et ordonancer des données 
from sklearn import datasets #modul de sklearn pour récuperer un jeux de données existant
from sklearn.ensemble import RandomForestClassifier #import du model de machine learning utilisé dans l'exemple
import streamlit as st #lib qui permet de générer l'interface web


#Ce fichier est un exemple d'application web simple pour la prédiction avec un model
#il n'est pas encore question d'explicabilité mais il représente le premier objectif 
#savoir se servir d'un model simple


#lib streamlit la fonction write permet d'écrire dans l'app du texte mais aussi de tableau etc 
#(ici il s'agit du titre et d'une phrase d'introduction)
st.write('''
# Bienvenue dans un exemple d'application de pédiction avec un model de machine learning
Il s'agit d'un exemple de l'interface elle sera ensuite agrémentée de plusieurs éléments d'éxplicabilités
''')

#lib streamlit sidebar(permet de scinder l'interface avec une barre latéral) header(set le titre du sidebar)
st.sidebar.header("Les parmaètres d'entrée")

#entrée: none
#sortie: DataFrame (from pandas) paramètres d'entrées set par l'utilisateur
#description: genère (pas dynamiquement) les sliders permetant d'ajuster la valeur des parmètres d'entrées 
#puis créé un dataframe avec ces même données
def param_entree():
    longeur_sepal= st.sidebar.slider('Longeur du sepal', 4.3, 7.9, 5.3)
    largeur_sepal= st.sidebar.slider('Largeur du sepal', 2.0, 4.4, 3.3)
    longeur_petal= st.sidebar.slider('Longeur du petal', 1.0, 6.9, 2.3)
    largeur_petal= st.sidebar.slider('Largeur du petal', 0.1, 2.5, 1.3)
    donnees={
        'longeur_sepal':longeur_sepal,
        'largeur_sepal' :largeur_sepal,
        'longeur_petal':longeur_petal,
        'largeur_petal':largeur_petal
    }
    fleur_parametres=pd.DataFrame(donnees,index=[0])
    return fleur_parametres

#appel de la fonction param_entree et stockage du dataframe dans df_entree
df_entree=param_entree()

st.subheader('on veut prédire la catégorie de cette fleur')
#affichage du dataframe (valeurs des paramètres d'entrées) dans l'interface 
st.write(df_entree)

#entrée: none
#sortie: DataFrame (from pandas) prédiction du model
#description: récupère un jeu de données exisant dans sklearn (iris) puis fais la prédiction avec les paramètres d'entrée,
#cette prédiction et mis dans un dataframe puis on le retourne
def param_sortie():
    iris=datasets.load_iris()
    clf=RandomForestClassifier()
    clf.fit(iris.data,iris.target)

    prediction=clf.predict(df_entree)
    donnees={
        'prediction':prediction,
        'nom_prediction':iris.target_names[prediction]#on récupère le nom de la fleur avec le chiffre prédit 
                                                    #(les noms sont dans un tableau déjà exisant dans le modul sklearn.dataset)
    }
    resultat_prediction=pd.DataFrame(donnees,index=[0])
    return resultat_prediction

df_sortie=param_sortie()   

st.subheader('Catégorie de la fleut prédit par le model')

#tableau statique qui liste les resultats possibles de la prédiction 
tableau_explicatif=pd.DataFrame({'setosa':0,'versicolor':1,'virginica':2},index=[0])

st.write('''Résultats possibles:''',tableau_explicatif,'''Prédiction:''',df_sortie)