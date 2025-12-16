"""
@ IOC - CE IABD
Script per analitzar el Port del Cantó amb clustering de ciclistes
"""
import os
import logging
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
import numpy as np

logging.basicConfig(level=logging.INFO)

# ----------------- Funcions -----------------

def load_dataset(path):
    """ Carrega el dataset de registres dels ciclistes """
    df = pd.read_csv(path)
    logging.info("Dataset carregat amb %d files", len(df))
    return df

def exploratory_data_analysis(df):
    """ Exploratory Data Analysis del dataframe """
    logging.info("Exploratory Data Analysis del dataframe")
    logging.info("\n%s", df.describe())
    logging.info(df.info())

def clean(df):
    """ Neteja de les dades: elimina id i tt """
    df_clean = df.drop(columns=['id', 'temps_total'])
    logging.info("Columnes 'id' i 'tt' eliminades")
    # Renombrar columnes per simplificar
    df_clean = df_clean.rename(columns={'temps_pujada':'tp', 'temps_baixada':'tb'})
    return df_clean

def extract_true_labels(df):
    """ Guardem les etiquetes dels ciclistes (tipus) """
    labels = df['tipus'].values
    logging.info("Labels reals extrets: %s", set(labels))
    return labels

def visualitzar_pairplot(df, path_img='img/pairplot.png'):
    """ Genera un pairplot dels atributs """
    os.makedirs(os.path.dirname(path_img), exist_ok=True)
    sns.pairplot(df)
    plt.savefig(path_img)
    plt.close()
    logging.info("Pairplot generat a %s", path_img)

def clustering_kmeans(data, n_clusters=4):
    """ Crea i entrena el model KMeans """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data)
    logging.info("Model KMeans entrenat")
    return model

def visualitzar_clusters(df, labels, path_img='img/clusters.png'):
    """ Visualitza els clusters amb colors diferents """
    df_plot = df.copy()
    df_plot['label'] = labels
    os.makedirs(os.path.dirname(path_img), exist_ok=True)
    sns.scatterplot(x='tp', y='tb', hue='label', palette='Set1', data=df_plot)
    plt.xlabel('temps pujada (tp)')
    plt.ylabel('temps baixada (tb)')
    plt.title('Clusters de ciclistes')
    plt.savefig(path_img)
    plt.close()
    logging.info("Clusters visualitzats i guardats a %s", path_img)

def associar_clusters_patrons(tipus, model):
    """ Associa els labels del clustering amb els patrons reals """
    logging.info("Centres dels clústers:")
    for j, center in enumerate(model.cluster_centers_):
        logging.info("%d:    (tp: %.1f     tb: %.1f)", j, center[0], center[1])

    # Assignació manual basada en suma tp+tb
    ind_label_0 = ind_label_1 = ind_label_2 = ind_label_3 = -1
    suma_max = 0
    suma_min = 50000

    for j, center in enumerate(model.cluster_centers_):
        suma = center[0] + center[1]
        if suma_max < suma:
            suma_max = suma
            ind_label_3 = j
        if suma_min > suma:
            suma_min = suma
            ind_label_0 = j

    tipus[0].update({'label': ind_label_0})
    tipus[3].update({'label': ind_label_3})

    lst = [0,1,2,3]
    lst.remove(ind_label_0)
    lst.remove(ind_label_3)

    if model.cluster_centers_[lst[0]][0] < model.cluster_centers_[lst[1]][0]:
        ind_label_1 = lst[0]
        ind_label_2 = lst[1]
    else:
        ind_label_1 = lst[1]
        ind_label_2 = lst[0]

    tipus[1].update({'label': ind_label_1})
    tipus[2].update({'label': ind_label_2})

    logging.info("Associació tipus i labels: %s", tipus)
    return tipus

def generar_informes(df, tipus):
    """ Genera fitxers d'informes per cada clúster """
    os.makedirs('informes', exist_ok=True)
    for t in tipus:
        label = t['label']
        nom = t['name']
        df_label = df[df['label'] == label]
        df_label.to_csv(f'informes/{nom}.txt', index=False)
    logging.info("S'han generat els informes en la carpeta informes/")

def nova_prediccio(dades, model):
    """ Classifica nous ciclistes """
    dades_array = np.array([[d[1], d[2]] for d in dades])  # només tp i tb
    preds = model.predict(dades_array)
    for i, d in enumerate(dades):
        logging.info("Nou ciclista id %d - tipus %s - classe %d", d[0], d[3], preds[i])
    return dades, preds

# ----------------- Main -----------------

if __name__ == "__main__":

    # Carregar dataset
    PATH_DATASET = './data/ciclistes.csv'
    df_ciclistes = load_dataset(PATH_DATASET)

    # Exploratory Data Analysis
    exploratory_data_analysis(df_ciclistes)

    # Neteja de dades
    df_clean_local = clean(df_ciclistes)

    # Extreure labels reals i eliminar columna tipus
    true_labels_local = extract_true_labels(df_ciclistes)
    df_clean_local = df_clean_local.drop(columns=['tipus'])

    # Visualitzar pairplot
    visualitzar_pairplot(df_clean_local)

    # Entrenar model KMeans
    clustering_model_local = clustering_kmeans(df_clean_local)

    # Guardar model
    os.makedirs('model', exist_ok=True)
    with open('model/clustering_model.pkl', 'wb') as f:
        pickle.dump(clustering_model_local, f)
        logging.info("Model KMeans guardat a model/clustering_model.pkl")

    # Scores
    hom_local = homogeneity_score(true_labels_local, clustering_model_local.labels_)
    comp_local = completeness_score(true_labels_local, clustering_model_local.labels_)
    vmes_local = v_measure_score(true_labels_local, clustering_model_local.labels_)
    scores_local = {'homogeneity': hom_local, 'completeness': comp_local, 'v_measure': vmes_local}
    with open('model/scores.pkl', 'wb') as f:
        pickle.dump(scores_local, f)
    logging.info("Scores calculats i guardats: %s", scores_local)

    # Visualitzar clusters
    visualitzar_clusters(df_clean_local, clustering_model_local.labels_)

    # Afegir columna label al dataframe
    df_clean_local['label'] = clustering_model_local.labels_

    # Associar labels amb patrons
    tipus_local = [{'name':'BEBB'}, {'name':'BEMB'}, {'name':'MEBB'}, {'name':'MEMB'}]
    tipus_local = associar_clusters_patrons(tipus_local, clustering_model_local)

    # Guardar tipus_dict.pkl
    with open('model/tipus_dict.pkl', 'wb') as f:
        pickle.dump(tipus_local, f)

    # Generar informes
    generar_informes(df_clean_local, tipus_local)

    # Classificació de nous ciclistes
    nous_ciclistes_local = [
        [500, 3230, 1430, 4660], # BEBB
        [501, 3300, 2120, 5420], # BEMB
        [502, 4010, 1510, 5520], # MEBB
        [503, 4350, 2200, 6550] # MEMB
    ]
    dades_local, preds_local = nova_prediccio(nous_ciclistes_local, clustering_model_local)
