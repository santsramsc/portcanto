""" @ IOC - CE IABD - MLflow KMeans ciclistes """

import os
import logging
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

# Importa les funcions del script clustersciclistes.py
from clustersciclistes import load_dataset, clean, extract_true_labels

logging.basicConfig(level=logging.INFO)

# ----------------- Preparació de dades -----------------

# Ruta del dataset
PATH_DATASET = './data/ciclistes.csv'

# Carregar i preparar dades
df_ciclistes = load_dataset(PATH_DATASET)
df_clean = clean(df_ciclistes)
true_labels = extract_true_labels(df_ciclistes)
df_clean = df_clean.drop(columns=['tipus'])  # eliminar columna de labels reals

# Convertir a array
X = df_clean.values
X_sample = X[:5]  # petit exemple per a input_example en MLflow

# ----------------- MLflow -----------------

# Crear l'experiment MLflow
mlflow.set_experiment("K sklearn ciclistes")

with mlflow.start_run(run_name="K_range") as run:

    for K in range(2, 9):
        logging.info("Entrenant KMeans amb K=%d", K)
        
        # Entrenar KMeans
        model = KMeans(n_clusters=K, random_state=42)
        model.fit(X)
        logging.info("Model KMeans entrenat")
        
        # Calcular mètriques
        hom = homogeneity_score(true_labels, model.labels_)
        comp = completeness_score(true_labels, model.labels_)
        vmes = v_measure_score(true_labels, model.labels_)
        logging.info("K=%d | hom=%.3f | comp=%.3f | v_measure=%.3f", K, hom, comp, vmes)
        
        # Registrar mètriques a MLflow
        mlflow.log_metric("homogeneity", hom, step=K)
        mlflow.log_metric("completeness", comp, step=K)
        mlflow.log_metric("v_measure", vmes, step=K)
        
        # Guardar model a MLflow
        mlflow.sklearn.log_model(model, name=f"KMeans_{K}", input_example=X_sample)

logging.info("S'han generat tots els runs a MLflow")
