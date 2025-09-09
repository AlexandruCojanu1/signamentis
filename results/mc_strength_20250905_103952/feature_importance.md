# Multiclass Strength - Feature Importance

- Thresholds: low=0.333852 (p=0.3), high=0.652586 (p=0.7)

## Global (LightGBM)

- **price_level_diff**: 5021.000000
- **m15_volume_avg_4**: 4049.000000
- **m15_current_range_percentile**: 3458.000000
- **m5_volume_avg**: 3314.000000
- **m15_volume_avg_8**: 3059.000000
- **m15_volume_trend**: 2246.000000
- **current_volume**: 2056.000000
- **m5_volume_trend**: 1980.000000
- **m15_vol_regime**: 1815.000000
- **m15_volume_ratio_4_8**: 1803.000000
- **m5_range_expansion**: 1593.000000
- **m5_support_distance**: 1444.000000
- **vol_m5_vs_m15**: 1416.000000
- **m15_range_volatility**: 1397.000000
- **m15_momentum_1h**: 1384.000000

## Global (RandomForest)

- **price_level_diff**: 0.258695
- **m15_volume_avg_4**: 0.076828
- **price_level_alignment**: 0.070113
- **m15_current_range_percentile**: 0.058522
- **m15_volume_avg_8**: 0.057584
- **m5_volume_avg**: 0.044198
- **trend_strength_ratio**: 0.042472
- **current_volume**: 0.034601
- **m15_avg_range**: 0.030909
- **m15_range_volatility**: 0.021748
- **m15_volume_trend**: 0.021392
- **m5_volatility_6**: 0.018985
- **m5_avg_range_3**: 0.018156
- **m5_avg_range_6**: 0.017850
- **m15_vol_regime**: 0.015823

## Class: STRONG

- **price_level_diff**: 1313.000000
- **m15_volume_avg_4**: 1209.000000
- **m15_current_range_percentile**: 973.000000
- **m5_volume_avg**: 852.000000
- **m15_volume_avg_8**: 826.000000
- **m15_volume_trend**: 537.000000
- **current_volume**: 444.000000
- **m15_vol_regime**: 394.000000
- **m5_volume_trend**: 356.000000
- **m15_volume_ratio_4_8**: 320.000000
- **m5_support_distance**: 308.000000
- **m5_range_expansion**: 278.000000
- **m15_range_volatility**: 263.000000
- **m15_momentum_1h**: 260.000000
- **m5_resistance_distance**: 258.000000

## Class: MEDIUM

- **price_level_diff**: 1807.000000
- **m15_volume_avg_4**: 1236.000000
- **m15_current_range_percentile**: 1048.000000
- **m15_volume_avg_8**: 805.000000
- **m5_volume_avg**: 690.000000
- **current_volume**: 490.000000
- **m15_volume_trend**: 454.000000
- **m5_volume_trend**: 318.000000
- **m15_momentum_1h**: 315.000000
- **m15_vol_regime**: 308.000000
- **m15_range_volatility**: 295.000000
- **m15_avg_range**: 294.000000
- **m5_support_distance**: 268.000000
- **m15_volume_ratio_4_8**: 263.000000
- **m5_range_expansion**: 249.000000

## Class: WEAK

- **price_level_diff**: 1270.000000
- **m15_volume_avg_4**: 1058.000000
- **m15_current_range_percentile**: 995.000000
- **m5_volume_avg**: 796.000000
- **m15_volume_avg_8**: 769.000000
- **current_volume**: 514.000000
- **m15_volume_trend**: 481.000000
- **m5_volume_trend**: 449.000000
- **m15_volume_ratio_4_8**: 392.000000
- **m15_avg_range**: 364.000000
- **m15_vol_regime**: 347.000000
- **m5_range_expansion**: 283.000000
- **m15_momentum_1h**: 282.000000
- **m15_trend_slope**: 267.000000
- **m15_range_volatility**: 258.000000

