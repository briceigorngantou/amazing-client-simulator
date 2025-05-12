import streamlit as st
import numpy as np
import requests
import uuid
from datetime import datetime

st.set_page_config(page_title="Simulateur Client Amazing", layout="centered")
st.title("🛍️ Simulateur de Client - Amazing Marketplace")
st.markdown("""
Ce formulaire permet de simuler un **nouveau client** en fonction de ses habitudes d'achat.
À partir de ces données, notre modèle IA prédit à **quelle catégorie** il appartient.
""")

# Génération automatique des IDs
user_id = np.random.randint(10000, 99999999)
product_id = np.random.randint(100000, 99999999)
category_id = np.random.randint(100, 99999)
user_session = str(uuid.uuid4())

# Sélecteurs de catégories imbriquées
st.header("🧾 Habitudes d'achat du client")
cat_lvl1 = st.selectbox("Catégorie principale du produit", ["apparel", "electronics", "home", "sports"])
cat_lvl2_options = {
    "apparel": ["clothing", "shoes", "accessories"],
    "electronics": ["smartphones", "laptops", "audio"],
    "home": ["furniture", "kitchen", "decor"],
    "sports": ["fitness", "outdoor", "team-sports"]
}
cat_lvl2 = st.selectbox("Sous-catégorie", cat_lvl2_options[cat_lvl1])

cat_lvl3_options = {
    "clothing": ["men", "women", "kids"],
    "shoes": ["sneakers", "boots", "sandals"],
    "accessories": ["bags", "watches", "jewelry"],
    "smartphones": ["android", "ios", "accessories"],
    "laptops": ["gaming", "ultrabook", "2-in-1"],
    "audio": ["headphones", "speakers", "soundbars"],
    "furniture": ["bedroom", "living-room", "office"],
    "kitchen": ["appliances", "cookware", "storage"],
    "decor": ["wall-art", "lighting", "plants"],
    "fitness": ["yoga", "gym", "running"],
    "outdoor": ["camping", "hiking", "biking"],
    "team-sports": ["football", "basketball", "tennis"]
}
cat_lvl3 = st.selectbox("Détail", cat_lvl3_options[cat_lvl2])
category_code = f"{cat_lvl1}.{cat_lvl2}.{cat_lvl3}"

# Marque
brand = st.selectbox("Marque", ["Nike", "Apple", "Samsung", "Adidas", "Sony", "Zara"])

# Prix
price = st.number_input("Prix (€)", min_value=1.0, step=0.1)

# Date et heure d'événement
# Date et heure de l'événement
date_input = st.date_input("Date de l'événement", datetime.utcnow().date())
time_input = st.time_input("Heure de l'événement", datetime.utcnow().time())
event_time = datetime.combine(date_input, time_input)

# Vues et achats
n_views = st.number_input("Nombre de vues", min_value=1, step=1)
n_purchases = st.number_input("Nombre d'achats", min_value=0, step=1)

# Prédiction
if st.button("Prédire la catégorie client"):
    payload = {
        "user_id": user_id,
        "product_id": product_id,
        "category_id": category_id,
        "category_code": category_code,
        "brand": brand,
        "price": price,
        "user_session": user_session,
        "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n_views": n_views,
        "n_purchases": n_purchases
    }

    st.json(payload)  # Affiche les données envoyées

    try:
        response = requests.post("http://localhost:8000/predict_label", json=payload)
        data = response.json()

        if "label" in data:
            st.success(f"✅ Ce client appartient au segment : **{data['label']}** (Cluster {data['final_cluster_label']})")
        else:
            st.error(f"❌ Erreur : {data.get('error', 'inconnue')}")
    except Exception as e:
        st.error(f"⚠️ Une erreur est survenue lors de la requête : {e}")

# Footer
st.markdown("""
---
**Projet IA & Big Data - 5e Année**  
Équipe Amazing 2025  
Yassine Mahfoudh  
Faikoth Achake Monrenike SALAMI  
NGANTOU YAMTCHEU Brice Igor  
""")
