#!/bin/bash
set -e

# get slug from pre-training
SLUG="dungsunf/autolab-exp-20261009-220044"

echo "=== 2. POST-TRAINING (160 epochs, dim=128, layers=3) ==="
mkdir -p /home/sunf/FSB/Final/Movie4U/multiautoresearch/post-training/dataset_upload/final_model
cp /home/sunf/FSB/Final/Movie4U/multiautoresearch/pre-training/.runtime/kaggle-logs/${SLUG/\//_}/experiment/final_model/gnn.pt /home/sunf/FSB/Final/Movie4U/multiautoresearch/post-training/dataset_upload/final_model/gnn.pt

cd /home/sunf/FSB/Final/Movie4U/multiautoresearch/post-training
uv run scripts/kaggle_job.py launch --mode experiment --train-args="--epochs 160 --hidden-channels 128 --num-layers 3 --lr 0.001" > launch_output_post.txt
SLUG_POST=$(cat launch_output_post.txt | grep "Kernel slug:" | awk -F "Kernel slug: " '{print $2}')
if [ -z "$SLUG_POST" ]; then
    SLUG_POST=$(grep -o "dungsunf/[a-zA-Z0-9-]*" launch_output_post.txt | head -n 1)
fi
echo "Launched post-training: $SLUG_POST"

while true; do
    STATUS=$(uv run scripts/kaggle_job.py status $SLUG_POST | grep "Status:" | awk -F "Status: " '{print $2}')
    if [ -z "$STATUS" ]; then
        STATUS=$(uv run scripts/kaggle_job.py status $SLUG_POST | grep "has status" | grep -o '"KernelWorkerStatus.[A-Z]*"' | tr -d '"' | cut -d'.' -f2)
    fi
    echo "Post-training status: $STATUS"
    if [[ "$STATUS" == "COMPLETE" || "$STATUS" == "complete" ]]; then
        break
    elif [[ "$STATUS" == "ERROR" || "$STATUS" == "error" || "$STATUS" == "failed" ]]; then
        echo "Post-training failed."
        exit 1
    fi
    sleep 30
done

echo "Downloading post-training output..."
uv run scripts/kaggle_job.py output $SLUG_POST

echo "Pipeline finished successfully!"
