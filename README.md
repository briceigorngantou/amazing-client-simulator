#  Amazing Client Segment Simulator

Ce projet Streamlit permet de simuler les habitudes d’achat d’un nouveau client et de prédire à quel segment il appartient, grâce à un modèle de Machine Learning (KMeans).

---

##  Prérequis

- Python 3.10 ou supérieur
- pip

---

##  Installation

### 1. Cloner ou extraire le projet dans un dossier
```bash
git clone https://github.com/Yassine-Mahfoudh/amazing-client-simulator.git
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv .venv
```

### 3. Activer l'environnement virtuel

#### Sous Windows :
```bash
.\venv\Scripts\activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```
##  Lancer l’application Streamlit

Dans le terminal :

```bash
streamlit run client_segment_app.py
```

Cela ouvrira une interface web dans le navigateur où vous pourrez remplir un formulaire simulant un nouveau client.

Installez FastAPI et Uvicorn :

```bash
pip install fastapi uvicorn scikit-learn==1.2.2
```

Lancer le serveur :

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```