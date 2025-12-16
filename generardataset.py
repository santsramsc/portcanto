"""
Script per generar el dataset de ciclistes amb dades sintètiques
"""

import os
import logging
import csv
import numpy as np

STR_CICLISTES = 'data/ciclistes.csv'

# Configuració bàsica de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

def generar_dataset(num, ind, dicc_aux):
    """
    Genera els temps dels ciclistes, de forma aleatòria, però en base a la 
    informació del diccionari
    TODO: completar arguments, return. num és el número de files/ciclistes a 
    generar. ind és l'index/identificador/dorsal.
    """
    dataset = []
    for i in range(num):
        tipus = dicc_aux[i % len(dicc_aux)]
        # ciclistes distribuïts de forma equilibrada
        temps_pujada = int(np.random.normal(tipus["mu_p"], tipus["sigma"]))
        # temps pujada amb distribució normal
        temps_baixada = int(np.random.normal(tipus["mu_b"], tipus["sigma"]))
        # temps baixada amb distribució normal
        temps_total = temps_pujada + temps_baixada
        dataset.append({
            "id": ind + i,
            "temps_pujada": temps_pujada,
            "temps_baixada": temps_baixada,
            "temps_total": temps_total,
            "tipus": tipus["name"]
        })
    return dataset

def guardar_csv(dataset, fitxer):
    """
    Guarda el dataset en un fitxer CSV.
    
    Args:
        dataset (list): llista de dicc_localionaris amb dades dels ciclistes
        fitxer (str): ruta del fitxer CSV de sortida
    """
    # crea la carpeta si no existeix
    os.makedirs(os.path.dirname(fitxer), exist_ok=True)
    with open(STR_CICLISTES, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "temps_pujada", "temps_baixada",
        "temps_total", "tipus"])
        writer.writeheader()  # escriu els noms de les columnes
        for fila in dataset:
            writer.writerow(fila)  # escriu cada ciclista

if __name__ == "__main__":

    # BEBB: bons escaladors, bons baixadors
    # BEMB: bons escaladors, mal baixadors
    # MEBB: mal escaladors, bons baixadors
    # MEMB: mal escaladors, mal baixadors

    # Port del Cantó (18 Km de pujada, 18 Km de baixada)
    # pujar a 20 Km/h són 54 min = 3240 seg
    # pujar a 14 Km/h són 77 min = 4268 seg
    # baixar a 45 Km/h són 24 min = 1440 seg
    # baixar a 30 Km/h són 36 min = 2160 seg
    MU_P_BE = 3240 # mitjana temps pujada bons escaladors
    MU_P_ME = 4268 # mitjana temps pujada mals escaladors
    MU_B_BB = 1440 # mitjana temps baixada bons baixadors
    MU_B_MB = 2160 # mitjana temps baixada mals baixadors
    SIGMA = 240 # 240 s = 4 min

    dicc_local = [
        {"name":"BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name":"BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name":"MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name":"MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]

    # Generar el dataset
    dataset_gen = generar_dataset(num=100, ind=1, dicc_aux=dicc_local)

    # Guardar el dataset a CSV
    guardar_csv(dataset_gen, STR_CICLISTES)

    logging.info("s'ha generat data/ciclistes.csv")
