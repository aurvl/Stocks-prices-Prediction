# Prédiction des Prix des Actions

## Description du Projet

Ce projet vise à prédire les prix des actions sur une période future de 30 jours en utilisant un modèle d'apprentissage automatique. L'application Streamlit fournit une interface utilisateur conviviale pour afficher les données historiques et les prédictions générées.

Le projet est structuré pour séparer les différentes étapes :
- **Training** : Préparation, entraînement du modèle et génération des fichiers nécessaires.
- **Deploy** : Scripts pour le déploiement et l'exécution de l'application interactive.

---

## Fonctionnalités

- **Prédiction** : Modèle basé sur un réseau de neurones récurrents (RNN) pour les prédictions.
- **Visualisation** : Interface interactive pour explorer les données historiques et futures.

---

## Structure du Répertoire

```
├── deploy/                         # Scripts pour le déploiement
│   ├── streamlit_app.py            # Application Streamlit interactive
│   ├── wpea_pred.py                # Script pour les prédictions quotidiennes
├── src/                            # Fichiers de base pour le modèle
│   ├── metrics.txt                 # Évaluation des performances du modèle
│   ├── preprocessor                # Scaler pour le prétraitement des données
│   ├── results.xlsx                # Résultats des prédictions
│   └── wpea_pred_model.h5          # Modèle entraîné (sauvegardé au format HDF5)
├── training/                       # Scripts et ressources pour l'entraînement
│   └── prediction_wpea_script.py   # Script d'entraînement et de prétraitement
│   └── functions.py                # Fonctions utilitaires pour l'entraînement
├── requirements.txt                # Dépendances Python nécessaires
└── Readme.md                       # Documentation du projet
```

---

## Installation

### Étape 1 : Cloner le dépôt
```bash
git clone https://github.com/aurvl/Stocks-prices-Prediction.git
cd Stocks-prices-Prediction
```

### Étape 2 : Créer un environnement virtuel
```bash
python3 -m venv my_virtual_env
```

### Étape 3 : Activer l'environnement virtuel
- **Windows** :
  ```bash
  my_virtual_env/Scripts/activate
  ```
- **Linux/Mac** :
  ```bash
  source env/bin/activate
  ```

### Étape 4 : Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## Utilisation

### 1. Entraînement du Modèle
Pour entraîner un modèle ou modifier le processus d'entraînement, utilisez :
```bash
python training/prediction_wpea_script.py
```

### 2. Exécution de l'Application Streamlit
Pour lancer l'application Streamlit :
```bash
streamlit run deploy/streamlit_app.py
```
L'application sera accessible à l'adresse `http://localhost:8501/`.

### 3. Exécution Automatisée des Prédictions
Pour générer des prédictions mises à jour (par exemple, chaque jour) :
```bash
python deploy/wpea_pred.py
```

---

## Modèle Utilisé

Le modèle est un **RNN (LSTM)** entraîné avec les données historiques téléchargées via **Yahoo Finance**. Les variables explicatives incluent :
- `day_of_week`, `week_of_year`, `month_of_year`, `quarter_of_year`, `semester_of_year` : Calendrier
- `lag_1_week` : Valeur retardée sur 5 jours
- `Vol_1_month` : Volume glissant sur 20 jours
- `SMA20`, `SMA50` : Moyennes mobiles sur 20 et 50 jours.
- `RSI` : Indice de tendance de 10 jours.
- `return` : Rendement quotidien.

---

## Visualisation

L'application Streamlit vous permet :
1. **D'explorer les données historiques** avec des graphiques interactifs.
2. **De visualiser les prédictions futures** en couleur distincte.

---

## Résultats des Prédictions

Les résultats des prédictions sont enregistrés dans le fichier `src/results.xlsx` pour un suivi et une analyse ultérieure.

---

## Contributions

Les contributions sont les bienvenues. Veuillez soumettre une *issue* ou une *pull request* pour discuter des améliorations.