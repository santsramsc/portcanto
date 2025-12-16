"""
@ IOC - CE IABD
Tests unitaris per generació de dataset i clustering de ciclistes
"""
import unittest
import os
import pickle

from generardataset import generar_dataset
from clustersciclistes import (
    load_dataset,
    clean,
    extract_true_labels,
    clustering_kmeans
)

# ---------- Constants ----------
MU_P_BE = 3240   # mitjana temps pujada bons escaladors
MU_P_ME = 4268   # mitjana temps pujada mals escaladors
MU_B_BB = 1440   # mitjana temps baixada bons baixadors
MU_B_MB = 2160   # mitjana temps baixada mals baixadors
SIGMA = 240      # desviació estàndard

DICC_CICLISTES = [
    {"name": "BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
    {"name": "BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
    {"name": "MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
    {"name": "MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
]

PATH_DATASET = "./data/ciclistes.csv"
PATH_MODEL = "./model/clustering_model.pkl"


# ---------- Tests P1 ----------
class TestGenerarDataset(unittest.TestCase):
    """
    Tests per la funció generar_dataset
    """

    def test_longituddataset(self):
        """Comprova que el dataset té la longitud correcta"""
        arr = generar_dataset(200, 1, DICC_CICLISTES)
        self.assertEqual(len(arr), 200)

    def test_valorsmitjatp(self):
        """Comprova que el temps mitjà de pujada és raonable"""
        arr = generar_dataset(100, 1, [DICC_CICLISTES[0]])
        temps_pujada = [row["temps_pujada"] for row in arr]
        mitjana_tp = sum(temps_pujada) / len(temps_pujada)
        self.assertLess(mitjana_tp, 3400)

    def test_valorsmitjatb(self):
        """Comprova que el temps mitjà de baixada és elevat per mals baixadors"""
        arr = generar_dataset(100, 1, [DICC_CICLISTES[1]])
        temps_baixada = [row["temps_baixada"] for row in arr]
        mitjana_tb = sum(temps_baixada) / len(temps_baixada)
        self.assertGreater(mitjana_tb, 2000)


# ---------- Tests P2 ----------
class TestClustersCiclistes(unittest.TestCase):
    """
    Tests per al clustering de ciclistes
    """

    @classmethod
    def setUpClass(cls):
        """Preparació comuna per tots els tests"""
        df = load_dataset(PATH_DATASET)
        df_clean = clean(df)
        cls.true_labels = extract_true_labels(df)
        df_clean = df_clean.drop(columns=["tipus"])

        cls.clustering_model = clustering_kmeans(df_clean)
        cls.data_labels = cls.clustering_model.labels_
        cls.df_clean = df_clean

        os.makedirs("model", exist_ok=True)
        with open(PATH_MODEL, "wb") as file:
            pickle.dump(cls.clustering_model, file)

    def test_check_column(self):
        """Comprova que existeix la columna tp"""
        self.assertIn("tp", self.df_clean.columns)

    def test_data_labels(self):
        """Comprova que el nombre de labels coincideix amb les dades"""
        self.assertEqual(len(self.data_labels), len(self.df_clean))

    def test_model_saved(self):
        """Comprova que el model s'ha guardat correctament"""
        self.assertTrue(os.path.isfile(PATH_MODEL))


if __name__ == "__main__":
    unittest.main()
