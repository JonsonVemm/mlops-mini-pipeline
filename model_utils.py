from huggingface_hub import hf_hub_download
import joblib
import time


def load_model(repo_id: str, filename: str = "model.pkl", force_download: bool = False):
    t0 = time.time()
    local_path = hf_hub_download(
        repo_id=repo_id, filename=filename, force_download=force_download
    )
    model = joblib.load(local_path)
    print(f"✅ Modelo carregado do Hub/Cache em {time.time() - t0:.3f}s")
    return model
