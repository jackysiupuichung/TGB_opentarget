#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=32G
#$ -l h_rt=24:0:0
#$ -cwd
#$ -j y

set -euo pipefail

echo "=== 🚀 Starting THGL OpenTargets Model Run ==="
date

# === Activate environment ===
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "📦 Activated .venv"
else
    echo "❌ ERROR: .venv not found in working directory"
    exit 1
fi

echo "📂 Working directory: $(pwd)"

# === Run Data Preprocessing ===
echo "🚀 Preprocessing THGL OpenTargets data..."
# uncomment this if you haven't generated the dataset yet
# python tgb/datasets/thgl_opentargets/thgl_opentargets.py

# === Run Negative Sampling Generation (Optional) ===
# Uncomment this if you haven't generated negative samples yet
# echo "🎲 Generating negative samples..."
# python tgb/datasets/thgl_opentargets/thgl_opentargets_ns_gen.py

# === Run Models ===
# Uncomment the model you wish to run

# 1. EdgeBank
echo "▶️ Running EdgeBank..."
python examples/linkproppred/thgl-opentargets/edgebank.py --data thgl-opentargets --mem_mode unlimited

# 2. TGN
# echo "▶️ Running TGN..."
# python examples/linkproppred/thgl-opentargets/tgn.py --num_epoch 50 --bs 200 --lr 0.0001 --tolerance 1e-4 --patience 5

# 3. Recurrency Baseline
# echo "▶️ Running Recurrency Baseline..."
# python examples/linkproppred/thgl-opentargets/recurrencybaseline.py --data thgl-opentargets

# 4. STHN
# echo "▶️ Running STHN..."
# python examples/linkproppred/thgl-opentargets/sthn.py --data thgl-opentargets

echo "🎉 Model run completed!"
date
echo "======================================="
