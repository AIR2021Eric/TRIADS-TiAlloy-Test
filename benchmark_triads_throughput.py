import os, sys, time, warnings
import torch
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from pymatgen.core import Composition
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Setup path
current_dir = os.getcwd()
repo_dir = os.path.join(current_dir, "TRIADS") if os.path.exists(os.path.join(current_dir, "TRIADS", "matbench_steels")) else current_dir
if repo_dir not in sys.path: sys.path.insert(0, repo_dir)
from matbench_steels.model import DeepHybridTRM, ExpandedFeaturizer

MAGPIE, EXTRA, M2V = 132, 200, 200
BATCH_SIZES = [1, 10, 100, 1000, 5000, 10000]
NUM_SAMPLES = 10000

def flatten(feat):
    if isinstance(feat, (list, tuple)) and len(feat) > 0 and np.ndim(feat[0]) >= 1: feat = feat[0]
    return np.asarray(feat, dtype=np.float32).ravel()

def fix_feature_vector(feat):
    if len(feat) == MAGPIE + EXTRA + M2V: return feat
    magpie, m2v = feat[:MAGPIE], feat[-M2V:]
    extra = np.pad(feat[MAGPIE:-M2V], (0, EXTRA - (len(feat) - MAGPIE - M2V)), constant_values=0.0)
    return np.concatenate([magpie, extra, m2v])

def main():
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

    print(f"\nFeaturizing {NUM_SAMPLES} dummy alloys (mix of steels and Ti-alloys)...")
    dummy_formulas = ["Fe0.7Cr0.15Ni0.15", "Ti90Al6V4", "Ti85V15", "Fe0.98C0.02", "Ti81Al9Mo2V5Cr3"] * (NUM_SAMPLES // 5)
    
    # --- THE FIX ---
    # Featurize one-by-one to avoid the inhomogeneous np.array() crash inside the author's code
    fixed_feats = []
    for f in tqdm(dummy_formulas, desc="Featurizing"):
        comp = Composition(f)
        # Pass as a list of 1 to satisfy the batch API, get the first element back
        raw_feat = featurizer.featurize_all([comp])[0] 
        fixed_feat = fix_feature_vector(flatten(raw_feat))
        fixed_feats.append(fixed_feat)
    # ---------------
        
    print("Stacking into tensor...")
    X_all = torch.tensor(np.array(fixed_feats), dtype=torch.float32)
    print(f"Total tensor shape: {X_all.shape}")

    results = []
    print("\n" + "="*60); print("RUNNING THROUGHPUT BENCHMARK (CPU)"); print("="*60)
    
    for bs in BATCH_SIZES:
        torch.set_num_threads(8) # Utilize CPU cores
        t0 = time.time()
        
        with torch.no_grad():
            for i in range(0, len(X_all), bs):
                batch = X_all[i:i+bs]
                # Run ensemble
                preds = [m(batch) for m in models]
                # Average the 5 folds
                ensemble_pred = torch.stack(preds).mean(dim=0)
                
        dt = time.time() - t0
        sps = NUM_SAMPLES / dt
        print(f"Batch Size {bs:>5}: {NUM_SAMPLES} samples in {dt:.3f}s -> {sps:,.1f} samples/sec")
        
        results.append({
            "model": "TRIADS (5-fold ensemble)",
            "hardware": "CPU (i7-14700HX)",
            "batch_size": bs,
            "total_samples": NUM_SAMPLES,
            "time_sec": round(dt, 3),
            "samples_per_sec": round(sps, 1)
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("triads_throughput_benchmark.csv", index=False)
    print("\n[Saved] triads_throughput_benchmark.csv")

if __name__ == "__main__":
    main()