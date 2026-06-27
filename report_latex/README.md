# LaTeX report

Figures và metrics nằm tại `../data/` (gitignored).

```bash
make -C myresearch/report_latex
```

Refresh figures sau khi render:

```bash
cp ../data/eval_python/view_*_pred_gt_err.png ../data/report_figures/python/
cp ../data/render_cuda/view_*_pred_gt_err.png ../data/render_cuda/view_*_pred.png \
   ../data/report_figures/cuda/
```

Requires: `pdflatex`, `babel-vietnamese` (tlmgr install babel-vietnamese vntex).
