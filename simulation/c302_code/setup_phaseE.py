# setup_phaseE.py
import os, json, yaml

base = "Projects/C-Elegans/PhaseE"

# Directory scaffold
dirs = [
    f"{base}/config",
    f"{base}/shared",
    f"{base}/idea1_signatures/outputs",
    f"{base}/idea2_neuropeptides/outputs",
    f"{base}/idea3_link_pred/outputs",
    f"{base}/idea4_gnn_seq2seq/outputs",
    f"{base}/idea5_perturbations/outputs",
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# ---- Shared config files ----
env_yaml = {
    "seed": 42,
    "device": "cuda",
    "epochs": {"link_pred": 100, "dynamics": 120},
    "patience": 12,
    "lr": 1e-3,
    "weight_decay": 1e-5,
}
thresholds_yaml = {
    "expr_min_tpm": 0.5,
    "co_presence_min_frac": 0.1,
    "peptide_edge_aggregation": "max",
    "strict": {"expr_z": 0.5, "min_pairs": 1},
    "relaxed": {"expr_z": 0.2, "min_pairs": 1},
}
neuron_list = []  # placeholder, you’ll populate with canonical 179 common_neurons

# Write configs if missing
env_path = f"{base}/config/env.yaml"
if not os.path.exists(env_path):
    with open(env_path, "w") as f:
        yaml.dump(env_yaml, f)

thr_path = f"{base}/config/thresholds.yaml"
if not os.path.exists(thr_path):
    with open(thr_path, "w") as f:
        yaml.dump(thresholds_yaml, f)

neurons_path = f"{base}/shared/neuron_lists.json"
if not os.path.exists(neurons_path):
    with open(neurons_path, "w") as f:
        json.dump(neuron_list, f, indent=2)

# ---- Shared utils stub ----
utils_path = f"{base}/shared/utils.py"
if not os.path.exists(utils_path):
    with open(utils_path, "w") as f:
        f.write(
            '"""\nShared utilities for PhaseE notebooks.\nFill in with actual functions later.\n"""\n\n'
            "def set_all_seeds(seed):\n    pass\n\n"
            "def load_common_neurons():\n    pass\n\n"
        )

print("✅ PhaseE scaffold created at", base)
