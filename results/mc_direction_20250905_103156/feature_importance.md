# Multiclass Direction - Feature Importance

- Strong threshold: |ret| ≥ 0.005240 (quantile=0.8)

## Global (LightGBM)

- **m5_price_momentum_3**: 3701.000000
- **price_level_diff**: 3132.000000
- **future_ret**: 2996.000000
- **m5_price_momentum_6**: 2247.000000
- **m5_volatility_3**: 1754.000000
- **m5_max_move_3**: 1401.000000
- **m5_volatility_6**: 1307.000000
- **m15_volume_ratio_4_8**: 1290.000000
- **m5_range_expansion**: 1101.000000
- **m5_support_distance**: 970.000000
- **m15_vol_regime**: 920.000000
- **m5_price_acceleration**: 887.000000
- **m15_volume_avg_8**: 787.000000
- **m5_volume_avg**: 684.000000
- **m5_volume_trend**: 682.000000

## Global (RandomForest)

- **future_ret**: 0.391100
- **price_level_diff**: 0.250028
- **m15_momentum_1h**: 0.071622
- **trend_strength_ratio**: 0.031301
- **m15_current_range_percentile**: 0.027221
- **price_level_alignment**: 0.022229
- **m15_volume_avg_4**: 0.021342
- **m5_price_momentum_3**: 0.019128
- **m15_volume_avg_8**: 0.014375
- **m15_vol_regime**: 0.011688
- **momentum_divergence**: 0.010937
- **m15_range_volatility**: 0.010235
- **momentum_confluence**: 0.010158
- **m15_trend_slope**: 0.010063
- **m5_volume_avg**: 0.009497

## Class: UP

- **m5_price_momentum_3**: 689.000000
- **future_ret**: 403.000000
- **m5_price_momentum_6**: 341.000000
- **price_level_diff**: 164.000000
- **m5_volatility_3**: 136.000000
- **m5_price_acceleration**: 86.000000
- **m5_max_move_3**: 73.000000
- **vol_m5_vs_m15**: 61.000000
- **m15_volume_ratio_4_8**: 51.000000
- **m5_volume_avg**: 48.000000
- **m5_resistance_distance**: 48.000000
- **m15_momentum_1h**: 44.000000
- **current_open**: 41.000000
- **m5_volume_trend**: 40.000000
- **m5_range_expansion**: 35.000000

## Class: SIDEWAYS

- **m5_price_momentum_3**: 693.000000
- **price_level_diff**: 400.000000
- **m5_price_momentum_6**: 321.000000
- **future_ret**: 213.000000
- **m5_volatility_3**: 161.000000
- **m5_price_acceleration**: 112.000000
- **m5_volatility_6**: 75.000000
- **m15_volume_ratio_4_8**: 69.000000
- **m5_range_expansion**: 66.000000
- **current_open**: 61.000000
- **m5_max_move_3**: 60.000000
- **m5_volume_trend**: 55.000000
- **m5_avg_range_6**: 54.000000
- **m5_avg_range_3**: 53.000000
- **m15_current_range_percentile**: 50.000000

## Class: DOWN

- **m5_price_momentum_3**: 589.000000
- **future_ret**: 406.000000
- **price_level_diff**: 371.000000
- **m5_price_acceleration**: 223.000000
- **m5_price_momentum_6**: 195.000000
- **m15_trend_slope**: 135.000000
- **m15_current_range_percentile**: 134.000000
- **m5_volume_trend**: 113.000000
- **m15_volume_avg_4**: 113.000000
- **m15_volume_ratio_4_8**: 102.000000
- **vol_m5_vs_m15**: 83.000000
- **m15_volume_avg_8**: 67.000000
- **current_open**: 66.000000
- **m5_volatility_3**: 58.000000
- **m5_max_move_3**: 55.000000

