# TRIADS Ti-Alloy OOD Stress-Test (Red-Team Project)
Role: red-team reviewer ONLY - NOT a primary screener.
Model: TRIADS (Tiwari 2026), matbench_steels head - HybridTRIADS V13A,
225K params, 5-fold ensemble, trained on 312 steel samples.
Sources: code = github.com/Rtx09x/TRIADS (clone into TRIADS/, gitignored);
weights = huggingface.co/Rtx09x/TRIADS -> weights/steels/weights.pt (runtime download).

## Setup
python -m venv venv ; activate ; pip install -r requirements_triads.txt
git clone https://github.com/Rtx09x/TRIADS.git   # required: architecture code
Mat2Vec embeddings download to mat2vec_cache/ on first featurizer run.

## Scripts
test_triads_steels.py            - 5-fold ensemble inference + OOD guardrails
benchmark_triads_throughput.py   - tensor-only batch sweep (1 -> 10000)
benchmark_triads_e2e.py          - end-to-end string -> matminer -> PyTorch (steels vs Ti)
plot_triads_e2e.py               - figure; reads CSVs + CrabNet peak dynamically
write_summary_triads.py          - exports ../comparison/model_results_triads.json

## Engineering fixes required to run the public checkpoint
1. HF path is weights/steels/weights.pt (not steels/weights.pt)
2. featurize_all is a BATCH API: pass a list of pymatgen Composition objects
3. Dimension drift: checkpoint pool.in=264 vs featurizer 532d
   (132 Magpie + 200 Extra + 200 Mat2Vec); n_extra solved from weights,
   input wiring probed (532 accepted)
4. matminer drops NaN columns for Ti/V/Mo -> 468d vectors; zero-pad the Extra
   block to restore 532d (Mat2Vec is always the trailing 200)

## Key findings (2026-09-24)
- Steel sanity: Fe0.7Cr0.15Ni0.15 -> 836.7 +/- 108.2 MPa (plausible, no flag)
- Ti-alloys -> ~830 MPa (regression to steel mean) -> OOD_STEEL_BIAS on 100% of Ti-rich inputs
- Tensor-only peak 3,783.5 samples/s @ batch 1000; E2E 27.3 (steels) / 26.0 (Ti) samples/s
  -> ~145x "feature engineering tax" from matminer descriptor computation
- Context: CrabNet E2E 244.33 samples/s (~9x faster than TRIADS E2E); Qwen best 1.711 samples/s
Verdict: fast in theory, matminer-bottlenecked in production, confidently wrong on
Ti-alloys; retained as OOD stress-test only.

## Outputs
triads_predictions.csv | triads_throughput_benchmark.csv | triads_e2e_benchmark.csv |
slide_triads_throughput_e2e.png | results_archive_triads/
