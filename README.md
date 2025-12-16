# Portcanto

Projecte desenvolupat per analitzar el Port del Cantó amb dades de ciclistes.
Inclou generació de dataset sintètic, clustering amb KMeans i seguiment amb MLflow.

## Contingut del projecte

- clustersciclistes.py → script principal per analitzar i clusteritzar els ciclistes.
- generardataset.py → genera el dataset sintètic de ciclistes.
- mlflowtracking-K.py → registre dels experiments amb MLflow.
- testportcanto.py → tests unitaris per verificar funcionament.
- docs/ → documentació en HTML del projecte.

## Instal·lació

1. Crear entorn virtual:

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Instal·lar dependències:

```powershell
pip install -r requirements.txt
```

## Ús

- Generar dataset: python generardataset.py
- Analitzar ciclistes: python clustersciclistes.py
- Executar tests: python -m unittest testportcanto.py

## Notes

- Evitar pujar la carpeta venv/ a GitHub.
- La documentació HTML està a la carpeta docs/.
