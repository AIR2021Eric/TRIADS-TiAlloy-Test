import os, sys, re, warnings
import torch
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from pymatgen.core import Composition

warnings.filterwarnings("ignore")

current_dir = os.getcwd()
repo_dir = (os.path.join(current_dir, "TRIADS")
            if os.path.exists(os.path.join(current_dir, "TRIADS", "matbench_steels"))
            else current_dir)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from matbench_steels.model import DeepHybridTRM, ExpandedFeaturizer

MAGPIE, EXTRA, M2V = 132, 200, 200   # block order printed by ExpandedFeaturizer

def flatten(feat):
    if isinstance(feat, (list, tuple)) and len(feat) > 0 and np.ndim(feat[0]) >= 1:
        feat = feat[0]
    return np.asarray(feat, dtype=np.float32).ravel()

def fix_feature_vector(feat):
    """
    Fixes the variable-length output caused by matminer dropping NaN columns 
    for elements like Ti/V. Pads the 'Extra' block with zeros to restore 532d.
    """
    if len(feat) == MAGPIE + EXTRA + M2V:
        return feat
    
    if len(feat) < MAGPIE + M2V:
        raise ValueError(f"Feature vector too short ({len(feat)}) to contain Magpie and Mat2Vec.")
        
    magpie = feat[:MAGPIE]
    m2v = feat[-M2V:]                  # Mat2Vec is always the last 200
    extra = feat[MAGPIE:-M2V]          # The middle block that lost columns
    
    if len(extra) < EXTRA:
        # Pad the missing Extra features with 0.0 to shift Mat2Vec back to its correct index
        extra = np.pad(extra, (0, EXTRA - len(extra)), constant_values=0.0)
        
    return np.concatenate([magpie, extra, m2v])

def load_ensemble():
    print("=" * 60); print("LOADING TRIADS STEELS ENSEMBLE (5-Fold)"); print("=" * 60)
    wpath = hf_hub_download(repo_id="Rtx09x/TRIADS", filename="weights/steels/weights.pt")
    ckpt = torch.load(wpath, map_location="cpu")
    featurizer = ExpandedFeaturizer()

    comp = Composition("Fe0.7Cr0.15Ni0.15")
    feat = flatten(featurizer.featurize_all([comp]))
    print(f"Featurizer output dim (Steel): {feat.shape[0]}")

    sd0 = ckpt['folds'][0]['model_state']
    exp_pool = sd0['pool.0.weight'].shape[1]
    Ec, cfg = ckpt.get('n_extra', 0), ckpt.get('config', {})
    Pc = DeepHybridTRM(n_extra=Ec, **cfg).pool[0].in_features
    Efix = exp_pool - (Pc - Ec)
    print(f"pool.in: checkpoint={exp_pool}, built(n_extra={Ec})={Pc} -> using n_extra={Efix}")

    models, wiring = [], None
    for i, fe in enumerate(ckpt['folds']):
        m = DeepHybridTRM(n_extra=Efix, **cfg)
        m.load_state_dict(fe['model_state'])
        m.eval()
        if wiring is None:
            probes = {}
            for d in sorted({exp_pool, MAGPIE + Efix + M2V, MAGPIE + EXTRA + M2V,
                             MAGPIE + Efix + EXTRA, 532}):
                try:
                    with torch.no_grad(): m(torch.zeros(1, d))
                    probes[d] = "OK"
                except Exception: probes[d] = "FAIL"
            print("input-dim probes:", probes)
            ok = [d for d, s in probes.items() if s == "OK"]
            prefer = [MAGPIE + Efix + M2V, exp_pool, MAGPIE + EXTRA + M2V, 532]
            wiring = next((d for d in prefer if d in ok), (ok[0] if ok else None))
            if wiring is None: raise RuntimeError("No accepted input dim found")
            print(f"selected wiring: input dim = {wiring}")
        models.append((m, wiring))
        print(f"  Loaded fold {i+1}/5 (test MAE: {fe.get('test_mae', 'N/A')} MPa)")
    return featurizer, models

def make_x(feat, dim):
    if dim == MAGPIE + EXTRA + M2V: return feat
    if dim <= MAGPIE + EXTRA: return feat[:dim]
    k = dim - MAGPIE - M2V
    return np.concatenate([feat[:MAGPIE], feat[MAGPIE:MAGPIE + k], feat[MAGPIE + EXTRA:]])

def predict(featurizer, models, formula):
    feat = flatten(featurizer.featurize_all([Composition(formula)]))
    
    # --- THE FIX ---
    original_len = len(feat)
    feat = fix_feature_vector(feat)
    if original_len != len(feat):
        print(f"  [Auto-Fix] Padded dropped matminer NaN columns: {original_len}d -> {len(feat)}d")
    # ---------------
    
    preds = []
    with torch.no_grad():
        for m, dim in models:
            x = torch.tensor(make_x(feat, dim)[None], dtype=torch.float32)
            preds.append(m(x).item())
    return float(np.mean(preds)), float(np.std(preds)), preds

def guardrails(formula, mean, std, preds):
    ti = re.search(r'Ti([0-9\.]+)', formula)
    ti_at = float(ti.group(1)) if ti else 0.0
    flags = ["OOD_STEEL_BIAS"] if ti_at > 50.0 else ["NONE"]
    if ti_at > 50.0:
        print("WARNING: Ti > 50 at% -> trained on 312 steels only -> OOD_STEEL_BIAS")
    sane = 0.0 < mean < 3000.0
    print(f"plausibility (0-3000 MPa): {'OK' if sane else 'SUSPECT'}")
    return {"formula": formula, "ys_MPa_mean": round(mean, 2), "ys_MPa_std": round(std, 2),
            "folds": [round(p, 2) for p in preds], "OOD_FLAGS": flags}

if __name__ == "__main__":
    featurizer, models = load_ensemble()
    results = []
    for alloy in ["Fe0.7Cr0.15Ni0.15", "Fe0.98C0.02", "Ti90Al6V4", "Ti85V15", "Ti81Al9Mo2V5Cr3"]:
        print(f"\n{'='*60}\nPREDICTING: {alloy}\n{'='*60}")
        try:
            mean, std, preds = predict(featurizer, models, alloy)
            print(f"Ensemble: {mean:.1f} +/- {std:.1f} MPa | folds: {[round(p,1) for p in preds]}")
            results.append(guardrails(alloy, mean, std, preds))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"formula": alloy, "error": str(e)})
    pd.DataFrame(results).to_csv("triads_predictions.csv", index=False)
    print("\n[Saved] triads_predictions.csv")