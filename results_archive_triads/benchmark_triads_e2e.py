import os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition
from tqdm import tqdm
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# Setup path
current_dir = os.getcwd()
repo_dir = os.path.join(current_dir, "TRIADS") if os.path.exists(os.path.join(current_dir, "TRIADS", "matbench_steels")) else current_dir
if repo_dir not in sys.path: sys.path.insert(0, repo_dir)
from matbench_steels.model import DeepHybridTRM, ExpandedFeaturizer

MAGPIE, EXTRA, M2V = 132, 200, 200
NUM_SAMPLES = 1000

def flatten(feat):
    if isinstance(feat, (list, tuple)) and len(feat) > 0 and np.ndim(feat[0]) >= 1: feat = feat[0]
    return np.asarray(feat, dtype=np.float32).ravel()

def fix_feature_vector(feat):
    if len(feat) == MAGPIE + EXTRA + M2V: return feat
    magpie, m2v = feat[:MAGPIE], feat[-M2V:]
    extra = np.pad(feat[MAGPIE:-M2V], (0, EXTRA - (len(feat) - MAGPIE - M2V)), constant_values=0.0)
    return np.concatenate([magpie, extra, m2v])

def load_system():
    print("="*60); print("LOADING TRIADS ENSEMBLE"); print("="*60)
    wpath = hf_hub_download(repo_id="Rtx09x/TRIADS", filename="weights/steels/weights.pt")
    ckpt = torch.load(wpath, map_location="cpu")
    featurizer = ExpandedFeaturizer()
    sd0 = ckpt['folds'][0]['model_state']
    exp_pool = sd0['pool.0.weight'].shape[1]
    Ec, cfg = ckpt.get('n_extra', 0), ckpt.get('config', {})
    Efix = exp_pool - (DeepHybridTRM(n_extra=Ec, **cfg).pool[0].in_features - Ec)
    models = []
    for fe in ckpt['folds']:
        m = DeepHybridTRM(n_extra=Efix, **cfg)
        m.load_state_dict(fe['model_state'])
        m.eval()
        models.append(m)
    return featurizer, models

def run_e2e_benchmark(formulas, label, featurizer, models):
    print(f"\n--- Benchmarking {label} ({NUM_SAMPLES} samples) ---")
    t0 = time.time()
    
    for f in tqdm(formulas, desc=label):
        comp = Composition(f)
        raw_feat = featurizer.featurize_all([comp])[0]
        fixed_feat = fix_feature_vector(flatten(raw_feat))
        x = torch.tensor(fixed_feat[None], dtype=torch.float32)
        with torch.no_grad():
            _ = torch.stack([m(x) for m in models]).mean().item()
            
    dt = time.time() - t0
    sps = NUM_SAMPLES / dt
    print(f"End-to-End Speed: {sps:.1f} samples/sec")
    return sps

if __name__ == "__main__":
    featurizer, models = load_system()
    
    # 1. Pure Steels (In-Domain)
    steels = ["Fe0.7Cr0.15Ni0.15", "Fe0.98C0.02", "Fe0.8Cr0.2", "Fe0.9Ni0.1", "Fe0.85Mn0.15"] * (NUM_SAMPLES // 5)
    # 2. Pure Ti-Alloys (OOD - The actual target domain)
    ti_alloys = ["Ti90Al6V4", "Ti85V15", "Ti81Al9Mo2V5Cr3", "Ti90V9Fe1", "Ti90V9Cr1"] * (NUM_SAMPLES // 5)
    
    sps_steel = run_e2e_benchmark(steels, "Pure Steels", featurizer, models)
    sps_ti = run_e2e_benchmark(ti_alloys, "Pure Ti-Alloys", featurizer, models)
    
    # --- SAVE TO CSV ---
    results_df = pd.DataFrame([
        {"scenario": "Pure Steels", "samples_per_sec": round(sps_steel, 2)},
        {"scenario": "Pure Ti-Alloys", "samples_per_sec": round(sps_ti, 2)}
    ])
    results_df.to_csv("triads_e2e_benchmark.csv", index=False)
    print("\n[Saved] triads_e2e_benchmark.csv")
    
    print("\n" + "="*50)
    print("VERDICT:")
    print(f"Steel Pipeline Speed: {sps_steel:.1f} samples/sec")
    print(f"Ti-Alloy Pipeline Speed: {sps_ti:.1f} samples/sec")