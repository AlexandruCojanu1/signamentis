# SOL/USDT 15m Direction Classifier

## Training Results
- Best Validation Accuracy: 0.3575
- Best Epoch: 3
- Macro F1 Score: 0.3357
- Temperature: 1.000

## Usage

### Training
```bash
python train_direction_15m_sol.py --epochs 50 --batch-size 256
```

### Inference
```bash
python infer_direction_15m_sol.py --save-dir ./checkpoints_direction_15m
```

## Files
- `model.pt`: Best model checkpoint
- `preproc_and_meta.json`: Preprocessing parameters and metadata
- `training_summary.json`: Complete training and evaluation results
- `training_metrics.csv`: Per-epoch training metrics
- `config.json`: Training configuration
