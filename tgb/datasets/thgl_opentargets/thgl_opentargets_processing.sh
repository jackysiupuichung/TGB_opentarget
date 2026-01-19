#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=8G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

set -euo pipefail

echo "=== 🚀 Starting THGL OpenTargets build job ==="
date

# === Activate environment ===
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "📦 Activated .venv"
else
    echo "❌ ERROR: .venv not found in working directory"
    exit 1
fi

# === Input directory ===
INPUT_DIR="/data/scratch/bty414/opentarget_evidences/23.06/progression_graph"
OUTPUT_DIR="/data/scratch/bty414/opentarget_evidences/23.06/thgl_opentargets"


echo "📂 Input directory:  $INPUT_DIR"
echo "📂 Output directory:  $OUTPUT_DIR"

# === Run THGL builder ===
echo "🚧 Building Temporal Heterogeneous Graph..."
python tgb/datasets/thgl_opentargets/thgl_opentargets.py \
    --dynamic_path "$INPUT_DIR/source_level_progression_dynamic.parquet" \
    --static_path "$INPUT_DIR/source_level_progression_static.parquet" \
    --out_dir "$OUTPUT_DIR"

# === Final status ===
echo
echo "🎉 THGL build completed successfully!"
date
echo "======================================="
