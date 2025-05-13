from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Charger modèles
with open("scaler.pkl", "rb") as f:
    scaler = joblib.load(f)
with open("pca.pkl", "rb") as f:
    pca = joblib.load(f)
with open("kmeans_model.pkl", "rb") as f:
    kmeans_final = joblib.load(f)

app = FastAPI()

# === Données entrées via formulaire ===
class EventInput(BaseModel):
    user_id: int
    product_id: int
    category_id: int
    category_code: str
    brand: str = None
    price: float
    user_session: str
    n_views: int
    n_purchases: int

# ===== Fonction d’analyse (simplifiée) =====
def predict_user_batch(df):
    from sklearn.preprocessing import LabelEncoder

    new_features = df.groupby('user_id').agg(
        total_events=('event_type', 'count'),
        n_sessions=('user_session', pd.Series.nunique),
        n_views=('event_type', lambda x: (x == 'view').sum()),
        n_purchases=('event_type', lambda x: (x == 'purchase').sum())
    ).reset_index()

    new_features['purchase_view_ratio'] = new_features['n_purchases'] / new_features['n_views']
    new_features['purchase_view_ratio'] = new_features['purchase_view_ratio'].fillna(0)

    views = df[df['event_type'] == 'view']
    purchases = df[df['event_type'] == 'purchase']

    avg_price_viewed = views.groupby('user_id')['price'].mean().rename('avg_price_viewed')
    avg_price_purchased = purchases.groupby('user_id')['price'].mean().rename('avg_price_purchased')
    max_price_purchased = purchases.groupby('user_id')['price'].max().rename('max_price_purchased')
    min_price_purchased = purchases.groupby('user_id')['price'].min().rename('min_price_purchased')

    new_features = new_features.merge(avg_price_viewed, on='user_id', how='left')
    new_features = new_features.merge(avg_price_purchased, on='user_id', how='left').fillna(0)
    new_features = new_features.merge(max_price_purchased, on='user_id', how='left').fillna(0)
    new_features = new_features.merge(min_price_purchased, on='user_id', how='left').fillna(0)

    new_features['n_unique_categories_viewed'] = views.groupby('user_id')['category_code'].nunique()
    new_features['n_unique_categories_purchased'] = purchases.groupby('user_id')['category_code'].nunique().fillna(0)

    for col in ['favorite_category_viewed', 'favorite_category_purchased', 'favorite_brand_viewed']:
        new_features[col] = 'unknown'

    for col in ['favorite_category_viewed', 'favorite_category_purchased', 'favorite_brand_viewed']:
        le = LabelEncoder()
        new_features[col] = le.fit_transform(new_features[col])

    new_features = new_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_scaled = scaler.transform(new_features.drop(columns=['user_id']))
    X_pca = pca.transform(X_scaled)[:, :2]
    cluster = kmeans_final.predict(X_pca)

    new_features['final_cluster_label'] = cluster
    return new_features


def generate_dynamic_cluster_labels(features_with_clusters):
    cluster_summary = features_with_clusters.groupby('final_cluster_label').agg({
        'purchase_view_ratio': 'mean',
        'avg_price_purchased': 'mean',
        'n_unique_categories_viewed': 'mean'
    }).reset_index()

    def label_cluster(row):
        ratio = row['purchase_view_ratio']
        avg_price = row['avg_price_purchased']
        n_cat = row['n_unique_categories_viewed']

        if ratio > 2 and avg_price > 2:
            return "VIP Clients"
        elif ratio > 1.5 and avg_price > 1.5:
            return "Gros Acheteurs"
        elif ratio > 1.5 and n_cat > 2:
            return "Acheteurs Curieux"
        elif ratio > 1:
            return "Acheteurs Réguliers"
        elif ratio < 0.5 and n_cat > 3:
            return "Explorateurs Intenses"
        elif ratio < 0.5 and n_cat > 1:
            return "Explorateurs"
        elif abs(ratio) < 0.3 and abs(avg_price) < 0.3 and abs(n_cat) < 0.3:
            return "Visiteurs Inactifs"
        else:
            return "Clients Classiques"

    cluster_summary['label'] = cluster_summary.apply(label_cluster, axis=1)
    return cluster_summary


@app.post("/predict_label")
def predict_label(event: EventInput):
    try:
        # Simule un mini-dataset à partir des entrées
        df = pd.DataFrame()

        # n_views lignes view, n_purchases lignes purchase
        for _ in range(event.n_views):
            df = pd.concat([df, pd.DataFrame([{
                "event_time": datetime.utcnow(),
                "event_type": "view",
                "product_id": event.product_id,
                "category_id": event.category_id,
                "category_code": event.category_code,
                "brand": event.brand,
                "price": event.price,
                "user_id": event.user_id,
                "user_session": event.user_session
            }])])

        for _ in range(event.n_purchases):
            df = pd.concat([df, pd.DataFrame([{
                "event_time": datetime.utcnow(),
                "event_type": "purchase",
                "product_id": event.product_id,
                "category_id": event.category_id,
                "category_code": event.category_code,
                "brand": event.brand,
                "price": event.price,
                "user_id": event.user_id,
                "user_session": event.user_session
            }])])

        # Feature engineering + clustering
        features = predict_user_batch(df)
        profiles = generate_dynamic_cluster_labels(features)
        final = features.merge(profiles, on="final_cluster_label", how="left")

        result = final[['user_id', 'final_cluster_label', 'label']].iloc[0].to_dict()
        return result

    except Exception as e:
        return {"error": str(e)}
