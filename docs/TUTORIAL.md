# Tutorial

Five short examples. The first runs on synthetic data with no download; the rest assume
you have loaded one hyperspectral cube into

```python
# X : (N, d) band-wise standardised pixels, row-major (N = H*W)
# y_true : (N,) ground truth, 0..C-1 on labelled pixels, -1 on background
# H, W, C : image height, width and number of classes
```

and drawn a partial-label vector `y` (a few labels per class, the rest `-1`). A seeded
`stratified_labels(y_true, n_per_class, seed)` helper is in `demo.py`; `unl` below is the
set of ground-truth pixels that were **not** revealed as labels:

```python
from demo import stratified_labels
y = stratified_labels(y_true, n_per_class=10, seed=42)
unl = (y_true >= 0) & (y < 0)
```

## 1. First run: synthetic scene, no data needed

```bash
python demo.py
```

```
Softmax      ACC=0.7424
SOP-SSFCM r=2 ACC=0.9549  (alpha=1539.9, 3 iterations)
label image: (64, 64) memberships U: (4096, 6) centroids V: (6, 30)
```

The two lines are the story of the paper in miniature: the spatially pooled prior gives
the gain over the raw posterior and the guided clustering adds a fuzzy partition.

## 2. One label budget on your own cube

```python
from sop_ssfcm import sop_ssfcm
from metrics import evaluate

res = sop_ssfcm(X, y, H, W, n_clusters=C, theta=0.99, r=2, seed=42)
print(evaluate(y_true[unl], res["labels"][unl]))     # {'acc', 'nmi', 'f1'}
```

## 3. Share one Softmax between variants

Training the Softmax is the expensive step; the pooled prior and the clustering are
cheap, so compare variants on the same posterior by passing `P`:

```python
from sop_ssfcm import train_softmax, posterior, sop_ssfcm
lab = y >= 0
W_, b = train_softmax(X[lab], y[lab], seed=42)
P = posterior(X, W_, b)
softmax_labels = P.argmax(1)
for r in (0, 1, 2):                                  # SOP-SSFCM (r = 0), SOP-SSFCM r=1, r=2
    res = sop_ssfcm(X, y, H, W, n_clusters=C, r=r, P=P)
    print(r, evaluate(y_true[unl], res["labels"][unl])["acc"], res["n_iter"])
```

## 4. θ, ω and the pooling operator

```python
from sop_ssfcm import theta_scales
ratio = theta_scales(X, y, seed=42)["ratio"]         # S_D/S_g measured once on the labelled pixels
for theta in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    res = sop_ssfcm(X, y, H, W, n_clusters=C, theta=theta, P=P, ratio=ratio)
    print(theta, round(res["alpha"], 1), round(res["share_g"], 3),
          evaluate(y_true[unl], res["labels"][unl])["acc"], res["n_iter"])
```

`share_g` (the guidance share realised at convergence) tracks θ; in the paper accuracy
increases monotonically with θ on every scene and the iteration count drops; at θ close
to 1 the labels follow the argmax of the pooled prior `π`. The pooling operator and ω:

```python
sop_ssfcm(..., pool="arith")          # linear pool: about +1.1 points of accuracy, but U no longer ranks reliability better than π
sop_ssfcm(..., omega=0.5)             # less weight on the neighbourhood than the default ω = 0.1
sop_ssfcm(..., omega=1.0)             # no neighbourhood, same as r = 0 (π = p)
```
