# Projet Business

### Structure du Répertoire

```
├── data/
│   ├── raw/                # Données brutes
│   └── processed/          # Données traitées
├── notebooks/              # Notebooks Jupyter
├── src/                    # Code source
│   ├── data_preprocessing/ # Scripts de prétraitement des données
│   ├── models/             # Scripts de modélisation
│   └── utils/              # Fonctions utilitaires
└── README.md               # Fichier README
```

## Description du Projet

Ce projet vise à analyser et modéliser des données commerciales pour extraire des insights pertinents et aider à la prise de décision.

## Objectif

L'objectif principal est de développer une application qui permet de visualiser et d'analyser les données commerciales afin de fournir des recommandations stratégiques.

## Dataset

Le dataset utilisé dans ce projet comprend des données de ventes, des informations sur les clients, et des données de marché. Les données sont stockées dans le répertoire `data`.

## Ressources Utilisées

- Python 3.11
- Pandas
- Numpy
- Tensorflow
- RNN

## Accéder au Produit Fini

Pour accéder à l'application finale, suivez les étapes ci-dessous :

1. Clonez le dépôt :
    ```bash
    git clone <URL_du_dépôt>
    ```
2. Créez un environnement virtuelle :
    ```bash
    python3 -m venv my_virtual_env
    ```
3. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```
4. Lancez l'application :
    ```bash
    streamlit run ./deploy/streamlit_app.py
    ```

L'application sera accessible via `http://localhost:5000`.