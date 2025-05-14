# update_data.py
import pandas as pd
import yfinance as yf
import gdown
from datetime import datetime

SYMBOL = "CW8.PA"
DRIVE_FILE_ID = "1eHvWadYri1NEZ2bRabsi6gyzyBAMeueO"  # À remplacer
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
LOCAL_PATH = "cw8_latest.csv"

def fetch_and_save():
    try:
        print("⏳ Téléchargement depuis Yahoo Finance...")
        data = yf.download(SYMBOL, progress=False)
        
        if data.empty:
            raise ValueError("Aucune donnée reçue")
        
        # Nettoyage des données
        data.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
                      for col in data.columns]
        data = data[data.index > '2018-04-17']
        
        # Sauvegarde locale
        data.to_csv(LOCAL_PATH)
        print("✅ Données sauvegardées localement")
        
        # Upload vers Drive
        gdown.download(DRIVE_URL, LOCAL_PATH, quiet=False)
        print("🔄 Données mises à jour sur Google Drive")
        
    except Exception as e:
        print(f"❌ Erreur : {str(e)}")

if __name__ == "__main__":
    print(f"\n🔍 Mise à jour du {datetime.now().strftime('%d/%m/%Y')}")
    fetch_and_save()