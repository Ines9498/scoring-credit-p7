import gdown
import os

def download_if_not_exists(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

def load_all_data():
    os.makedirs("data/original", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    
    files_to_download = {
        "data/original/bureau.csv": "https://drive.google.com/uc?id=10nsO7bWHEZ5Qkp6FwJpNNv30Sk5VvT0f",
        "data/original/bureau_balance.csv": "https://drive.google.com/uc?id=1-5tXP806KgKLwY409ZN5pJPVwClqyjtP",
        "data/original/credit_card_balance.csv": "https://drive.google.com/uc?id=1r2QmN3K2Xl3Ljzwy9wpceVfIXW3J1KEf",
        "data/original/installments_payments.csv": "https://drive.google.com/uc?id=1WDB6K3Erick8qWtyaYiDWP8iWoYmpUM9",
        "data/original/previous_application.csv": "https://drive.google.com/uc?id=1OFrwafw8OpH09oUZx2k8h-7HScdDg1a5",
        "data/original/POS_CASH_balance.csv": "https://drive.google.com/uc?id=1zughYQsfdnxEqGhZb1rBooYiQomDPdCv",
    }

    for path, url in files_to_download.items():
        download_if_not_exists(url, path)
