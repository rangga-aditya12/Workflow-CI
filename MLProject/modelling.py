import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Set path agar aman saat dijalankan di CI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'steam_data_clean_ready_preprocessing.csv')

# 1. Load Data
print("[INFO] Loading data...")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"[ERROR] File tidak ditemukan di: {DATA_PATH}")
    exit()

df = df.head(1500) # Sample 5k biar proses CI cepet

# 2. Set Experiment (Opsional di CI, tapi bagus buat log)
mlflow.set_experiment("CI_Steam_Experiment")

# 3. Training
print("[INFO] Training Model di GitHub Actions...")
mlflow.sklearn.autolog()

with mlflow.start_run():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # 4. Simpan Model sebagai Artifact Fisik
    # Ini penting buat Skilled: Kita simpan file-nya biar bisa dicommit balik ke GitHub
    model_name = "cosine_model_ci.pkl"
    save_path = os.path.join(BASE_DIR, model_name)
    
    with open(save_path, "wb") as f:
        pickle.dump(cosine_sim, f)
    
    mlflow.log_artifact(save_path)
    print(f"[SUCCESS] Model disimpan di: {save_path}")