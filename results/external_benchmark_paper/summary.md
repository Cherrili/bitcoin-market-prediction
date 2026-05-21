# Paper-Protocol Benchmark (Next-Day Direction)

## Disclaimer

Inspired by Omole & Enke (2024) (*Financial Innovation*, Boruta + CNN-LSTM, reported
**82.44%** max accuracy). This run uses **our merged CSV features** (not Glassnode) and
the same **80/20 chronological split** spirit, but is **not** a byte-for-byte replication.

## Task (paper-aligned)

- Class 1 if price(t+1) > price(t); Class 0 otherwise (Omole & Enke 2024)
- Date span: optional filter `2013-02-06` – `2023-02-18`

## Protocol vs. our 30-day benchmark

| Item | 30-day benchmark | This script (paper protocol) |
|------|------------------|------------------------------|
| Label | 30-day forward return | **Next-day** price direction |
| Test window | Strict partition only | **Standard**: lags may use train history |
| Window sizes | 30 only | **3, 5, 7, 14, 30** |

## Leakage controls

1. Next-day label computed then **removed** from features; only `label` kept for training.
2. Boruta / scaler fit on **train rows only**.
3. **Honest** row: window chosen by **validation F1** on train tail (no test peeking).
4. **Paper-style max** row: best test accuracy over window grid — matches how paper
   reports **overall max** across configurations (optimistic; not a single pre-registered test).

## Feature selection

- Method: BorutaPy; expanded to top-40 RF features
- Selected features: 40

## Results snapshot

```
               protocol  seed  window    split  accuracy       f1  precision   recall  roc_auc  n_samples               model
        grid_per_window    42     3.0 val_tail  0.557847 0.557679   0.558535 0.558724 0.570479      873.0                 NaN
        grid_per_window    42     3.0     test  0.508941 0.342228   0.367379 0.493508 0.493917      727.0                 NaN
        grid_per_window    42     5.0 val_tail  0.562428 0.561046   0.561026 0.561089 0.568793      873.0                 NaN
        grid_per_window    42     5.0     test  0.508941 0.367276   0.466143 0.494466 0.482689      727.0                 NaN
        grid_per_window    42     7.0 val_tail  0.553265 0.549190   0.550176 0.549518 0.555176      873.0                 NaN
        grid_per_window    42     7.0     test  0.507565 0.408916   0.487318 0.495136 0.488485      727.0                 NaN
        grid_per_window    42    14.0 val_tail  0.565865 0.562892   0.563389 0.562933 0.572331      873.0                 NaN
        grid_per_window    42    14.0     test  0.507565 0.362281   0.452509 0.492958 0.482576      727.0                 NaN
        grid_per_window    42    30.0 val_tail  0.554410 0.548486   0.550737 0.549481 0.574451      873.0                 NaN
        grid_per_window    42    30.0     test  0.510316 0.347686   0.423310 0.495015 0.478652      727.0                 NaN
        grid_per_window     0     3.0 val_tail  0.532646 0.475126   0.522545 0.514445 0.540689      873.0                 NaN
        grid_per_window     0     3.0     test  0.491059 0.491035   0.491760 0.491769 0.476697      727.0                 NaN
        grid_per_window     0     5.0 val_tail  0.563574 0.563388   0.564189 0.564402 0.569409      873.0                 NaN
        grid_per_window     0     5.0     test  0.506190 0.345766   0.388951 0.491015 0.487871      727.0                 NaN
        grid_per_window     0     7.0 val_tail  0.531501 0.491091   0.521731 0.516296 0.543773      873.0                 NaN
        grid_per_window     0     7.0     test  0.502063 0.496113   0.499095 0.499125 0.490939      727.0                 NaN
        grid_per_window     0    14.0 val_tail  0.567010 0.566710   0.571370 0.570713 0.569636      873.0                 NaN
        grid_per_window     0    14.0     test  0.511692 0.343471   0.399802 0.496174 0.483307      727.0                 NaN
        grid_per_window     0    30.0 val_tail  0.561283 0.559839   0.559822 0.559869 0.571240      873.0                 NaN
        grid_per_window     0    30.0     test  0.507565 0.346407   0.398818 0.492348 0.491136      727.0                 NaN
        grid_per_window     1     3.0 val_tail  0.537228 0.474792   0.530005 0.518206 0.548248      873.0                 NaN
        grid_per_window     1     3.0     test  0.514443 0.448135   0.507073 0.503981 0.494311      727.0                 NaN
        grid_per_window     1     5.0 val_tail  0.545246 0.528187   0.539589 0.535677 0.550977      873.0                 NaN
        grid_per_window     1     5.0     test  0.514443 0.361286   0.494610 0.499451 0.496280      727.0                 NaN
        grid_per_window     1     7.0 val_tail  0.536082 0.509287   0.528442 0.523827 0.551723      873.0                 NaN
        grid_per_window     1     7.0     test  0.510316 0.359249   0.461283 0.495451 0.492409      727.0                 NaN
        grid_per_window     1    14.0 val_tail  0.553265 0.548783   0.550018 0.549239 0.554873      873.0                 NaN
        grid_per_window     1    14.0     test  0.510316 0.340385   0.339921 0.494754 0.500492      727.0                 NaN
        grid_per_window     1    30.0 val_tail  0.554410 0.553811   0.554036 0.554227 0.563330      873.0                 NaN
        grid_per_window     1    30.0     test  0.513067 0.341607   0.382261 0.497420 0.502386      727.0                 NaN
        grid_per_window     7     3.0 val_tail  0.562428 0.561364   0.569011 0.567510 0.581036      873.0                 NaN
        grid_per_window     7     3.0     test  0.507565 0.336679   0.255895 0.492000 0.484061      727.0                 NaN
        grid_per_window     7     5.0 val_tail  0.571592 0.571587   0.573439 0.573497 0.579152      873.0                 NaN
        grid_per_window     7     5.0     test  0.510316 0.337887   0.256570 0.494667 0.479061      727.0                 NaN
        grid_per_window     7     7.0 val_tail  0.564719 0.564691   0.566208 0.566320 0.584428      873.0                 NaN
        grid_per_window     7     7.0     test  0.515818 0.340290   0.257909 0.500000 0.480871      727.0                 NaN
        grid_per_window     7    14.0 val_tail  0.561283 0.560061   0.560047 0.560149 0.583280      873.0                 NaN
        grid_per_window     7    14.0     test  0.515818 0.340290   0.257909 0.500000 0.489356      727.0                 NaN
        grid_per_window     7    30.0 val_tail  0.565865 0.565125   0.565235 0.565445 0.579392      873.0                 NaN
        grid_per_window     7    30.0     test  0.513067 0.339091   0.257241 0.497333 0.486455      727.0                 NaN
        grid_per_window    21     3.0 val_tail  0.549828 0.548974   0.555526 0.554515 0.564200      873.0                 NaN
        grid_per_window    21     3.0     test  0.508941 0.337284   0.256233 0.493333 0.493439      727.0                 NaN
        grid_per_window    21     5.0 val_tail  0.556701 0.556468   0.560623 0.560156 0.563320      873.0                 NaN
        grid_per_window    21     5.0     test  0.514443 0.342216   0.424263 0.498754 0.493364      727.0                 NaN
        grid_per_window    21     7.0 val_tail  0.561283 0.561262   0.562847 0.562941 0.563636      873.0                 NaN
        grid_per_window    21     7.0     test  0.513067 0.339091   0.257241 0.497333 0.486356      727.0                 NaN
        grid_per_window    21    14.0 val_tail  0.561283 0.561053   0.561749 0.561963 0.563457      873.0                 NaN
        grid_per_window    21    14.0     test  0.511692 0.345913   0.423630 0.496261 0.488061      727.0                 NaN
        grid_per_window    21    30.0 val_tail  0.555556 0.554667   0.554731 0.554889 0.566797      873.0                 NaN
        grid_per_window    21    30.0     test  0.514443 0.342216   0.424263 0.498754 0.491273      727.0                 NaN
    honest_val_selected    42    14.0     test  0.507565 0.362281   0.452509 0.492958 0.482576      727.0                 NaN
paper_style_max_on_test     7     7.0     test  0.515818      NaN        NaN      NaN      NaN        NaN                 NaN
            baseline_lr    42     NaN     test  0.503439 0.412519   0.480205 0.491485 0.497189      727.0 Logistic Regression
```

### Honest estimate (val-selected window, seed=42)

- Window = **14**
- Test accuracy = **0.5076** (paper max = 0.8244, gap = 0.3168)

### Paper-style grid maximum (exploratory)

- Best test accuracy over windows on same split = **0.5158** (window=7)
- Still **below 82.44%** with our data pipeline unless future work imports Glassnode features
  and full paper hyperparameter search.

## Interpretation

If honest accuracy remains near 50–60%, the gap to 82.44% is driven by **data source**,
**feature set**, and **reporting** (multi-seed/window max), not only model code.
Next-day labels are easier than 30-day horizons but do not automatically reproduce paper
numbers on a different feature matrix.

---
*Generated by `experiments/external_benchmark_paper_protocol.py`*
