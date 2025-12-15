import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# 1. Load Data
print("[INFO] Loading data...")
# Pastikan file csv sudah ada di folder yang sama
try:
    df = pd.read_csv('steam_data_clean_ready_preprocessing.csv')
except FileNotFoundError:
    print("[ERROR] File csv tidak ditemukan! Pastikan sudah dicopy ke folder ini.")
    exit()

# Sample data biar laptop gak nge-lag (10k data)
df = df.head(10000) 

# 2. Set Experiment MLflow
mlflow.set_experiment("Steam_Recommender_Basic")

# 3. Training dengan Autolog (Syarat Basic)
print("[INFO] Training Model (Basic)...")

# Aktifkan Autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Run_Autolog"):
    # Vektorisasi
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Hitung Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    print("[INFO] Model selesai dilatih.")
    
    # Simpan Model manual (Pickle) karena Cosine Sim bukan model standar sklearn
    with open("cosine_model.pkl", "wb") as f:
        pickle.dump(cosine_sim, f)
    
    # Log artifact manual
    mlflow.log_artifact("cosine_model.pkl")

print("[SUCCESS] Selesai! Cek MLflow UI.")