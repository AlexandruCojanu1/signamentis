.PHONY: mc-train
mc-train:
	python scripts/train_multiclass_direction.py --features data/processed/multi_timeframe_features_crypto_15m.csv --out_dir models/mc_direction --strong_quantile 0.8

.PHONY: mc-strength
mc-strength:
	python scripts/train_multiclass_strength.py --features data/processed/multi_timeframe_features_crypto_15m.csv --out_dir models/mc_strength --p_low 0.30 --p_high 0.70
