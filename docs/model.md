# Response assembly and evaluation

The query unit is one animal–target population. Matched non-targeting controls are split into disjoint halves in the original assay. Half 0 supplies query control information and half 1 defines the evaluation contrast `KO − NTC half 1`. The fixed joint pilot used a zero reference term at prediction time. Its training label preprocessing used a separate cross-fitted nuisance background.

## Response families

- `background` (`B`) is the common reference term.
- `U` is the low-rank response on the entire gene axis.
- `C` is the RNA change associated with class composition. Training label construction reweights control class expression by the observed perturbed composition.
- `W` is the remaining population RNA change after subtracting the composition-associated term on the adjusted training scale. It can also reflect error in estimating composition.

The calibrated mixture is `B + (1 − λ) × a × U + λ × (b × C + c × W)`. The component scales and mixture weight use the fixed grid `{0, 0.25, 0.5, 0.75, 1}`. Calibration consumes one completed, correctly aligned set of inner out-of-fold predictions. The same-input comparator is the strongest single family selected on those inner rows.

## Data format

NPZ files use Unicode identity arrays and finite real-valued numeric matrices. Loaders disable pickle. `source` and `target` each contain one identity per row. The combination `(source, target)` is unique across rows. `genes` is the ordered gene axis. `background`, `U`, `C`, `W`, and `truth` all have shape `(n_populations, n_genes)`. `truth` is the observed RNA change and is present only for calibration and scoring. Align all query arrays to the same frozen gene axis before running `predict`.

```bash
strokeperturb calibrate inner_oof.npz calibration.json
strokeperturb predict outer_components.npz forecasts.npz --calibration calibration.json
strokeperturb score outer_with_truth.npz scores.json --calibration calibration.json
```

`predict` applies the frozen coefficients. `score` independently assembles the same fixed forecasts and evaluates them against `truth`. The commands take explicit input paths. The surrounding experiment must build valid upstream fits and isolate scoring outcomes during model development.

For direct Python use, `decompose_training_bulk`, `benchmark_component_bridge`, and `split_training_residual` construct C/W labels and adjust their reference. `predict_linear_residual` implements the fixed linear, unpooled residual kernel and accepts optional U projection and decoding bases. Callers provide scope-qualified RNA and control data, the cross-fitted nuisance term, training-derived descriptor and reliability inputs, and training-derived U bases. The module docstrings specify each axis and the prediction inputs.

## Holdouts and endpoints

For the joint design, outer query rows have both held-out animal and held-out target identities. Training excludes every row sharing either held-out identity. Rows sharing only one of the held-out identities are excluded from both sets. This rule also applies at the inner boundary when producing calibration predictions.

The downstream endpoint is mean squared RNA error excluding the row's perturbed gene. The within-animal-centred endpoint additionally subtracts per-gene within-animal means from both truth and forecast. Centring uses the supported, individually masked entries. The summary first averages population losses within each target and then gives each target equal weight. See `strokeperturb/calibration.py` for the mask and tie rules.

The original internal pilot selected `B = 0`, `a = b = 0`, `c = 0.5`, and `λ = 1`. In that fold the mixture and the selected full-RNA family were identical. Subspace projection in the public Norman analysis is a separate fold-local operation on the fitted residual slope. The H9 interpolation baseline and the observed stroke contrasts also use separate input contracts and endpoints.

See the [benchmark tables](../reproduce/results/README.md) for the prespecified 1% comparison and the [joint-holdout diagnostics](../reproduce/joint_holdout/README.md) for the fixed two-mouse evaluation.
