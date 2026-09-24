import json, pandas as pd

thr = pd.read_csv('triads_throughput_benchmark.csv')
e2e = pd.read_csv('triads_e2e_benchmark.csv')
pred = pd.read_csv('triads_predictions.csv')

summary = {
  "model": "TRIADS (Steels Yield Strength)",
  "role": "Red-team reviewer (OOD stress-test)",
  "hardware": "CPU (i7-14700HX), 5-fold ensemble (225K params)",
  "theoretical_throughput_samples_per_s": {str(int(r.batch_size)): r.samples_per_sec for r in thr.itertuples()},
  "e2e_throughput_samples_per_s": {r.scenario: r.samples_per_sec for r in e2e.itertuples()},
  "best_batch_size_theoretical": 1000,
  "peak_theoretical_throughput": 3783.5,
  "ood_evidence": [
    "Predicts ~830 MPa for BOTH steels and Ti-alloys (regresses to training mean)",
    "Blind to Ti-physics; OOD_STEEL_BIAS flag applied to 100% of Ti-rich inputs",
    "End-to-End speed bottlenecked by matminer feature engineering (~26 samples/sec)"
  ],
  "verdict": "Extremely fast in theory but heavily bottlenecked by matminer in E2E. Confidently wrong on Ti-alloys; used to validate guardrails."
}
json.dump(summary, open(r'..\comparison\model_results_triads.json', 'w'), indent=2)
print('summary written to ..\\comparison\\model_results_triads.json')
