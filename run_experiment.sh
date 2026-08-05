#!/bin/bash
set -e

echo "=========================================="
echo "    V1G vs V2G Fleet Experiment Runner    "
echo "=========================================="

mkdir -p logs
export PYTHONPATH=$(pwd)

EPOCHS=5000

echo ""
echo ">>> [1/5] Training V1G Baseline (Smart Charging)..."
python3 -m scheduler -c configs/chicago_v1g.yaml -a TRAIN -o logs/v1g_train.csv --epochs $EPOCHS
mv ppo_policy.pt v1g_policy.pt

echo ""
echo ">>> [2/5] Evaluating V1G Baseline..."
python3 -m scheduler -c configs/chicago_v1g.yaml -a EVAL -p DNN -w v1g_policy.pt -o logs/v1g_eval.csv

echo ""
echo ">>> [3/5] Training V2G (Bidirectional Charging)..."
python3 -m scheduler -c configs/chicago_v2g.yaml -a TRAIN -o logs/v2g_train.csv --epochs $EPOCHS
mv ppo_policy.pt v2g_policy.pt

echo ""
echo ">>> [4/5] Evaluating V2G Agent..."
python3 -m scheduler -c configs/chicago_v2g.yaml -a EVAL -p DNN -w v2g_policy.pt -o logs/v2g_eval.csv

echo ""
echo ">>> [5/5] Generating Comparison Plots..."
# We will compare the charging power distributions
python3 -m analysis \
    -l logs/v1g_eval.csv logs/v2g_eval.csv \
    -f 50 --dt 3600 \
    --plot-charge-power-distribution

echo "Experiment Complete! Results logged to logs/"
