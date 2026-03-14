# DMD Experiment Notebook — Overview

This document explains **`dmd_experiment.ipynb`** in detail of the workflow and how to run or modify the experiments.

---

## 1. Purpose and context

The notebook implements a **One-Per-Env** baseline for the LEADS project (*Learning Dynamical Systems that Generalize Across Environments*, NeurIPS 2021). It:

- Uses **Dynamic Mode Decomposition (DMD)**-style methods from the [PyDMD](https://pydmd.github.io/PyDMD/) library.
- Compares **Exact DMD**, **BOP-DMD** (Optimized/Bagged DMD), and **EDMD** (Extended DMD with kernel) on two cheap-to-simulate systems:
  - **Lotka–Volterra (LV)** — 2D ODE (prey–predator).
  - **Gray–Scott (GS)** — 2D PDE (reaction–diffusion).
- Trains **one model per environment** (no sharing across envs).
- Evaluates **reconstruction** (on training data) and **forecast from new initial conditions** (test trajectories), and compares test MSE to the reported LEADS numbers.

Fixed random seeds (`np.random.seed(10)` for data/ICs, `99` for test ICs) keep results reproducible.

---

## 2. Prerequisites and setup

- **Python**: 3.7+ (recommended 3.10).
- **Install** (from repo root):
  ```bash
  pip install -r requirements.txt
  ```
  The notebook uses in particular: `numpy`, `matplotlib`, `scipy`, `pydmd`, `pandas`, `seaborn`.

- **Run the notebook**: open `dmd_experiment.ipynb` in Jupyter or VS Code and run cells in order. The first code cell imports:
  - `numpy`, `matplotlib`, `scipy.integrate.solve_ivp`, `functools.partial`, `pandas`
  - `from pydmd import DMD, BOPDMD, EDMD`
  - `from pydmd.plotter import plot_eigs`
  - `from numpy.fft import fft, fftfreq`
  - `seaborn`

---

## 3. Notebook structure (section by section)

### 3.1 LEADS baselines (reference numbers)

- Prints the **reported LEADS test MSE** (mean ± std) for LV and GS. These are the numbers the DMD baselines are compared to later (e.g. in the comparison table at the end).

### 3.2 Data generation (§1)

- **Lotka–Volterra**
  - Two **environments** = two parameter sets (`lv_params`: different `alpha`).
  - One **training trajectory** per env: 20 steps, `t ∈ [0, 10]`, `dt = 0.5`, same IC `y0` (from `np.random.seed(10)`).
  - State: `(prey, predator)`; snapshot matrix for PyDMD has shape `(2, n_steps)` (rows = state dimension, columns = time).

- **Gray–Scott**
  - Two **environments** = two parameter sets (`gs_params`: different `F`, `k`).
  - Grid: `32×32`, `dx = 1.0`; time horizon 400, evaluated every 40 → 10 time steps.
  - One **training trajectory** per env: state is `(u, v)` of shape `(2, n, 32, 32)`; for DMD it is reshaped to `(2*32*32, n)` (space flattened, columns = time).
  - Same IC for all envs: `u0 = 0.95`, `v0 = 0.05` with three 2×2 “square” perturbations (positions fixed by seed 10).

Important variables:

- `lv_trajs`, `gs_trajs`: list of arrays (one per env), **clean** trajectories.
- `lv_times`, `gs_times`: time points used for evaluation.

### 3.3 Noise injection (§2)

- **Training data** for DMD is **noisy**: Gaussian noise (σ = 0.01, seed 10) is added to the single training trajectory per env.
- Resulting lists: `lv_trajs_noisy`, `gs_trajs_noisy`. Clean `lv_trajs` / `gs_trajs` are kept for computing reconstruction MSE and as “ground truth” in plots.

### 3.4 Model fitting (§3) — One per environment

For each environment, **three** models are fitted **independently** on that env’s noisy training data only (One-Per-Env strategy).

**Lotka–Volterra**

- Data: `X_lv_noisy` (2×n_steps). For BOPDMD, data is **normalized** (zero mean, unit variance per row); FFT is used to get an initial `omega` for BOP-DMD eigenvalue guess.
- Models:
  - **DMD**: `DMD(svd_rank=2)`, `.fit(X_lv_noisy)`.
  - **BOPDMD**: `BOPDMD(svd_rank=2, num_trials=50, ...)` with `eig_constraints={"imag", "conjugate_pairs"}`, fitted on **normalized** data and `t = lv_times`.
  - **EDMD**: `EDMD(svd_rank=3, kernel_metric="rbf", kernel_params={"gamma": 0.5})`, `.fit(X_lv_noisy)`.
- Stored in `models_lv`: list of dicts, one per env; each dict has keys `"DMD"`, `"BOPDMD"`, `"EDMD"`, `"X_mean"`, `"X_std"`.

**Gray–Scott**

- Data: snapshot matrix `X` of shape `(2*32*32, n)` from the noisy trajectory; again **normalized** for BOPDMD; time for BOPDMD is **normalized** (`t_norm = gs_times / max(gs_times)`).
- Models:
  - **DMD**: `DMD(svd_rank=4)`, `.fit(X)`.
  - **BOPDMD**: `BOPDMD(svd_rank=2, ...)` on normalized `X` and `t_norm`.
  - **EDMD**: `EDMD(svd_rank=6, kernel_metric="rbf", kernel_params={"gamma": 0.5})`, `.fit(X)`.
- Stored in `models_gs`: list of dicts with `"DMD"`, `"BOPDMD"`, `"EDMD"`, `"X_mean"`, `"X_std"`, `"X_clean"`.

**In-sample reconstruction**

- LV: `reconstructed_data` (and for BOPDMD, denormalize with `X_mean`, `X_std`) → compare to clean `lv_trajs` and plot phase portrait + time series.
- GS: same idea; reconstruction MSE vs `X_clean` is printed.

There is also a small “Find best gamma” cell that sweeps EDMD RBF `gamma` for LV to show a reasonable default (0.5).

### 3.5 Visualization (§4)

- **Gray–Scott**: custom **3×3 summary** per model via `plot_gs_summary(...)`:
  - Row 1: mode amplitudes, discrete-time eigenvalues, continuous-time eigenvalues.
  - Rows 2–3: top 3 spatial modes (U component) and their time dynamics.
- For BOPDMD, modes are **denormalized** for display; eigenvalues are converted between discrete/continuous depending on model type.

### 3.6 Evaluation (§5)

- **Test trajectories**: 3 per env, **new initial conditions** (from `np.random.seed(99)` for LV; for GS, different ICs generated the same way as training but with seed 99).
- **Forecast from new IC**: the notebook uses a small helper **`forecast_dmd(model, x0_test)`**:
  - Projects `x0_test` onto the model’s modes to get new amplitudes `b_new`.
  - Uses the model’s **time evolution** (dynamics) with these amplitudes to predict the trajectory: `x(t) ≈ Φ (Ω(t) · b_new)`.
  - PyDMD does not provide this “forecast from new initial condition” API; `BOPDMD.forecast(t)` predicts at times `t` with the **fitted** amplitudes only.
- For **BOPDMD**, test ICs are **normalized** before calling `forecast_dmd`, and the predicted trajectory is **denormalized** before MSE.
- **Metrics**:
  - **Reconstruction MSE**: mean squared error between model reconstruction (on training time grid) and **clean** training data (printed per env for LV and GS).
  - **Test MSE**: mean squared error between **forecast from test IC** and the true test trajectory, averaged over test trajectories and envs.
- A **comparison table** and optional bar chart compare DMD / BOPDMD / EDMD test MSE with the reported LEADS test MSE.

---

## 4. Key variables and shapes (cheat sheet)

| Variable            | Meaning |
|---------------------|--------|
| `lv_trajs[e]`       | Clean LV trajectory for env `e`, shape `(2, 20)` |
| `lv_trajs_noisy[e]` | Noisy LV training data for env `e` |
| `gs_trajs[e]`       | Clean GS trajectory for env `e`, shape `(2, 10, 32, 32)` |
| `models_lv[e]["DMD"]` etc. | Fitted DMD/BOPDMD/EDMD object for LV env `e` |
| `models_gs[e]["DMD"]` etc. | Fitted DMD/BOPDMD/EDMD object for GS env `e` |
| `models_*[e]["X_mean"]`, `["X_std"]` | Normalization for BOPDMD (denormalize predictions) |

PyDMD convention: snapshot matrix **columns = time**, **rows = state** (flattened for GS).

---

## 5. Custom helpers (not in PyDMD)

- **`forecast_dmd(model, x0_test)`**  
  Returns predicted trajectory for a **new** initial condition `x0_test` using the fitted modes and dynamics (same length as `model.dynamics`). Used for test-set evaluation. PyDMD’s `BOPDMD.forecast(t)` only predicts at times `t` with the **fitted** amplitudes.

- **`plot_gs_summary(model, name, gs_env, gs_times, gs_dt_eval, gs_size, X_mean=None, X_std=None)`**  
  Draws the 3×3 Gray–Scott summary (amplitudes, eigenvalues, spatial modes, dynamics). Handles BOPDMD discrete/continuous eigenvalues and denormalization for display.

---

## 6. How to experiment

- **Change environments**: edit `lv_params` / `gs_params` (add or remove dicts). Adjust loops `range(len(lv_params))` / `range(len(gs_params))` accordingly.
- **Change training length**: e.g. `lv_times`, `gs_times` or time horizon / step; keep snapshot matrix shape consistent with `svd_rank` and model choices.
- **Noise level**: change `sigma` in the noise-injection cell (e.g. 0.01 → 0.05).
- **Seeds**: `np.random.seed(10)` for data/ICs, `99` for test ICs; change for different splits or ICs.
- **DMD variants**:
  - **DMD**: `svd_rank` (e.g. 2 for LV, 4 for GS).
  - **BOPDMD**: `svd_rank`, `num_trials`, `eig_constraints`, `init_alpha`, `varpro_opts_dict` (and for GS, normalized time).
  - **EDMD**: `svd_rank`, `kernel_metric`, `kernel_params` (e.g. `gamma` for RBF); the “Find best gamma” cell can be re-run for another range.
- **Test set size**: `n_test_trajs = 3`; increase for more stable average test MSE.
- **Comparison to LEADS**: the printed LEADS baseline (first cell) and the final comparison table assume the same evaluation protocol (test MSE on unseen ICs). Changing envs or test setup will change the baseline comparison.

---

## 7. References

- **LEADS**: [arXiv:2106.04546](https://arxiv.org/abs/2106.04546) — Learning Dynamical Systems that Generalize Across Environments.
- **PyDMD**: [pydmd.github.io](https://pydmd.github.io/PyDMD/) — DMD, BOPDMD, EDMD and plotting utilities.
