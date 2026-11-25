# Report figures

| File | Figure | Needs |
|---|---|---|
| `fig_overview.tex` | Figure 1 (overview) | `pdflatex fig_overview.tex` |
| `make_fig_fair.py` | Figure 2 (matched comparison on CIFAR-10) | matplotlib only; the numbers are written in the script |
| `make_fig_uci.py` | Figures 3 and 4 (UCI) | the saved runs in `../uci_regression/results/` (`stepC`, `stepD`, `stepF`) |

```bash
python make_fig_fair.py     # writes fig_fair_comparison.pdf
python make_fig_uci.py      # writes fig_origin.pdf and fig_changes.pdf
```

Note on Figure 2: the means and the AUROC standard deviations are the values reported in
the text. Three FPR@95 standard deviations (MSP in both settings, MDS without MixUp) were
read from the original plot and are approximate.
