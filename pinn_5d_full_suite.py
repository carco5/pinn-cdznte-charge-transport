# -*- coding: utf-8 -*-
# pinn_5d_full_suite.py
# PINN 5D (t,x,y,x0,y0): Drift + Diffusion + Trapping + Mass
# Planar geometry + uniform field + loss curriculum
# + early stopping + suite of 16 loss combinations (experiments)
# + pi-test (Welch) and full diagnostics (electron) for the FULL model.
#
# Adapted for execution on the university GPU:
#  - Force CUDA backend and check nvidia-smi.
#  - Matplotlib "Agg" backend if there is no DISPLAY.
#  - Save all figures in ablation/figs/<RUN_TAG> (PNG + PDF).
#  - Save metrics summary and pi-test in text/JSON files.

import os
import time
import subprocess
from typing import Any, Dict, Tuple, Optional
import numpy as np

# ========= 0) Paths and output folder =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIGS_ROOT = os.path.join(SCRIPT_DIR, "ablation", "figs")
FIGS_ROOT = os.environ.get("PINN_FIGS_DIR", DEFAULT_FIGS_ROOT)
RUN_TAG   = os.environ.get("PINN_RUN_TAG", time.strftime("abl_%Y%m%d_%H%M%S"))

# ========= 0.1) Minimal CUDA/JAX (GPU required) =========
def _cmd_ok(cmd: str) -> bool:
    return subprocess.run(
        ["bash", "-lc", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    ).returncode == 0

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_ID", "0"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["JAX_PLATFORM_NAME"] = "gpu"

if not _cmd_ok("nvidia-smi -L"):
    raise SystemExit("❌ GPU not visible. Run on a GPU node with CUDA.")

import jax
import jax.numpy as jnp
import jax.scipy.stats as js_stats  # (not strictly required, but we keep it)
import importlib.metadata as im

print(f"[JAX] jax={im.version('jax')} jaxlib={im.version('jaxlib')}")
_ = (
    jax.random.normal(jax.random.PRNGKey(0), (1024, 1024))
    @ jax.random.normal(jax.random.PRNGKey(1), (1024, 1024)).T
).block_until_ready()
print("✅ GPU OK")

import equinox as eqx
import optax

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from itertools import product

# ========= 0.2) Figure management (save to disk and show at the end) =========
FIGS = []

def add_fig(fig):
    """Register a figure to be saved at the end."""
    FIGS.append(fig)

def save_fig(fig):
    """Store a figure in the global buffer (without closing it, so it can be shown)."""
    fig.tight_layout()
    add_fig(fig)
    # We do NOT close it here; it will be saved + can be shown at the end


def finalize_figures(figs_root: str = FIGS_ROOT, run_tag: str = RUN_TAG):
    """Save all accumulated figures in PNG + PDF and (if applicable) display them."""
    from matplotlib.backends.backend_pdf import PdfPages
    import tarfile
    import matplotlib.pyplot as plt
    import matplotlib

    os.makedirs(figs_root, exist_ok=True)
    outdir = os.path.join(figs_root, run_tag)
    os.makedirs(outdir, exist_ok=True)

    # Individual PNGs
    for i, fig in enumerate(FIGS, 1):
        fig.savefig(
            os.path.join(outdir, f"fig_{i:03d}.png"),
            dpi=150,
            bbox_inches="tight"
        )

    # PDF with all figures
    pdf_path = os.path.join(outdir, "all_figs.pdf")
    with PdfPages(pdf_path) as pdf:
        for fig in FIGS:
            pdf.savefig(fig, bbox_inches="tight")

    # "latest" symlink and LAST_RUN
    try:
        latest = os.path.join(figs_root, "latest")
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(outdir, latest)
    except Exception:
        pass

    with open(os.path.join(figs_root, "LAST_RUN.txt"), "w") as f:
        f.write(os.path.abspath(outdir) + "\n")

    # Optional tarball
    if os.environ.get("PINN_MAKE_TAR", "0") == "1":
        tgz = os.path.join(figs_root, f"{run_tag}.tgz")
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(outdir, arcname=os.path.basename(outdir))

    print(f"[FIGS] {len(FIGS)} figures saved in {os.path.abspath(outdir)}/ (PNG + all_figs.pdf)")

    # Show all figures if backend is not Agg
    try:
        if matplotlib.get_backend().lower() != "agg":
            import matplotlib.pyplot as plt
            plt.show()
    except Exception:
        pass


# ==========================
# 1) Geometry in memory
# ==========================

GEOMETRY = {
    "dimensions": {
        "x": [0.0, 3.0e-3],
        "y": [0.0, 5.0e-3],
        "steps": [200, 150],
    },
    "expansion": {
        "pixels_per_side": 40
    },
    "physics": {
        "dt": 1.0e-9,
        "sim_time": 1.6e-6,
        "mu_e": 0.05,
        "mu_h": 0.0,
        "q": 1.0,
        "D_e": 1.0e-4,
        "tau_e": 2.0e-6,
    },
    "electrodes": {
        "anodes":   [{"position": {"x":[0.0,3.0e-3], "y":[0.0,0.0]}, "value": 0}],
        "drifts":   [],
        "cathodes": [{"position": {"x":[0.0,3.0e-3], "y":[5.0e-3,5.0e-3]}, "value": 450}],
    }
}

def _compute_uniform_field(electrodes: Dict[str, Any], y_bounds: Tuple[float, float]):
    y_min, y_max = y_bounds
    anodes = (electrodes or {}).get("anodes", []) or []
    cathodes = (electrodes or {}).get("cathodes", []) or []
    V_bot = anodes[0].get("value", 0.0) if anodes else 0.0
    V_top = cathodes[0].get("value", 0.0) if cathodes else 0.0
    Ex = 0.0
    Ey = - (float(V_top) - float(V_bot)) / max(1e-12, (y_max - y_min))
    return Ex, Ey

def load_geometry_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    dims = data["dimensions"]; physics = data.get("physics", {}) or {}; electrodes = data.get("electrodes", {}) or {}
    nx, ny = int(dims["steps"][0]), int(dims["steps"][1])
    x_min, x_max = float(dims["x"][0]), float(dims["x"][1])
    y_min, y_max = float(dims["y"][0]), float(dims["y"][1])
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = float(physics.get("dt", 1e-9))
    sim_time = float(physics.get("sim_time", 1.0e-6))
    mu_e = float(physics.get("mu_e", 0.0))
    D_e = float(physics.get("D_e", 0.0))
    tau_e = float(physics.get("tau_e", 0.0))
    q = float(physics.get("q", 1.0))
    Ex, Ey = _compute_uniform_field(electrodes, (y_min, y_max))
    vx, vy = mu_e * Ex, mu_e * Ey  # m/s
    return {
        "nx": nx, "ny": ny,
        "x_bounds": (x_min, x_max), "y_bounds": (y_min, y_max),
        "dx": dx, "dy": dy,
        "dt": dt, "t_bounds": (0.0, sim_time), "sim_time": sim_time,
        "mu_e": mu_e, "D_e": D_e, "tau_e": tau_e, "q": q,
        "Ex": Ex, "Ey": Ey, "vx": vx, "vy": vy,
        "electrodes": electrodes,
    }

def expand_domain(cfg_orig: Dict[str, Any], pad_px: int = 40) -> Dict[str, Any]:
    nx, ny = int(cfg_orig["nx"]), int(cfg_orig["ny"])
    dx, dy = float(cfg_orig["dx"]), float(cfg_orig["dy"])
    x0, x1 = cfg_orig["x_bounds"]; y0, y1 = cfg_orig["y_bounds"]
    nxE, nyE = nx + 2*pad_px, ny + 2*pad_px
    x0E, x1E = x0 - pad_px*dx, x1 + pad_px*dx
    y0E, y1E = y0 - pad_px*dy, y1 + pad_px*dy
    cfg_exp = dict(cfg_orig)
    cfg_exp.update({
        "nx": nxE, "ny": nyE,
        "x_bounds": (x0E, x1E), "y_bounds": (y0E, y1E),
        "dx": dx, "dy": dy, "pad_px": pad_px,
        "bounds_original": (cfg_orig["x_bounds"], cfg_orig["y_bounds"]),
    })
    cfg_exp["x_grid_exp"] = jnp.linspace(x0E, x1E, nxE)
    cfg_exp["y_grid_exp"] = jnp.linspace(y0E, y1E, nyE)
    return cfg_exp

def prepare_phi_and_grad(cfg_exp: Dict[str, Any]) -> Dict[str, Any]:
    Ex, Ey = float(cfg_exp["Ex"]), float(cfg_exp["Ey"])
    cfg_exp = dict(cfg_exp)
    cfg_exp["grad_phi_x"] = jnp.full((cfg_exp["nx"], cfg_exp["ny"]), -Ex)  # E = -∇φ
    cfg_exp["grad_phi_y"] = jnp.full((cfg_exp["nx"], cfg_exp["ny"]), -Ey)
    return cfg_exp

# ==========================
# 2) Normalizations and samplings
# ==========================

def _norm_01(v, vmin, vmax):
    return (v - vmin) / (vmax - vmin + 1e-30)

def normalize_txyx0y0(t: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray,
                      x0: jnp.ndarray, y0: jnp.ndarray,
                      cfg_xy: Dict, cfg_xy0: Optional[Dict] = None):
    cfg_xy0 = cfg_xy if cfg_xy0 is None else cfg_xy0
    t_max = float(cfg_xy["sim_time"])
    x_min, x_max = cfg_xy["x_bounds"]; y_min, y_max = cfg_xy["y_bounds"]
    x0_min, x0_max = cfg_xy0["x_bounds"]; y0_min, y0_max = cfg_xy0["y_bounds"]
    tN = t / (t_max if t_max > 0 else 1.0)
    xN, yN   = _norm_01(x,  x_min,  x_max), _norm_01(y,  y_min,  y_max)
    x0N,y0N  = _norm_01(x0, x0_min, x0_max), _norm_01(y0, y0_min, y0_max)
    return tN, xN, yN, x0N, y0N

def sample_ic_points_param(cfg_sample_xy: Dict, cfg_norm_xy: Dict, cfg_xy0: Dict,
                           N: int, *, key: jax.Array, sigma: float):
    (x0b,x1b),(y0b,y1b) = cfg_sample_xy["x_bounds"], cfg_sample_xy["y_bounds"]
    kx0, ky0, kx, ky = jax.random.split(key, 4)
    x0s = jax.random.uniform(kx0, (N,), minval=x0b, maxval=x1b)
    y0s = jax.random.uniform(ky0, (N,), minval=y0b, maxval=y1b)
    xs  = jax.random.uniform(kx,  (N,), minval=x0b, maxval=x1b)
    ys  = jax.random.uniform(ky,  (N,), minval=y0b, maxval=y1b)
    ts  = jnp.zeros_like(xs)

    tN, xN, yN, x0N, y0N = normalize_txyx0y0(ts, xs, ys, x0s, y0s, cfg_norm_xy, cfg_xy0)
    txyx0y0 = jnp.stack([tN, xN, yN, x0N, y0N], axis=1)

    s2 = sigma * sigma
    targ = jnp.exp(-(((xs - x0s)**2 + (ys - y0s)**2) / (2.0*s2)))
    return txyx0y0, xs, ys, x0s, y0s, targ

def sample_pde_points_param(cfg_xy: Dict, cfg_xy0: Dict, N: int, *,
                            key: jax.Array, t_max_phys: float):
    (x0b,x1b),(y0b,y1b) = cfg_xy["x_bounds"], cfg_xy["y_bounds"]
    (sx0,sx1),(sy0,sy1) = cfg_xy0["x_bounds"], cfg_xy0["y_bounds"]
    kt, kx, ky, kx0, ky0 = jax.random.split(key, 5)
    eps_t = 1e-9 * max(1.0, t_max_phys)
    ts  = jax.random.uniform(kt,  (N,), minval=eps_t, maxval=t_max_phys)
    xs  = jax.random.uniform(kx,  (N,), minval=x0b, maxval=x1b)
    ys  = jax.random.uniform(ky,  (N,), minval=y0b, maxval=y1b)
    x0s = jax.random.uniform(kx0, (N,), minval=sx0, maxval=sx1)
    y0s = jax.random.uniform(ky0, (N,), minval=sy0, maxval=sy1)
    tN, xN, yN, x0N, y0N = normalize_txyx0y0(ts, xs, ys, x0s, y0s, cfg_xy, cfg_xy0)
    return jnp.stack([tN, xN, yN, x0N, y0N], axis=1)

def sample_bc_anode_points_param(cfg_xy: Dict, cfg_norm_xy: Dict, cfg_xy0: Dict,
                                 N_space: int, N_times: int, *,
                                 key: jax.Array, t_max_phys: float):
    (x0b,x1b),(y0b,_) = cfg_xy["x_bounds"], cfg_xy["y_bounds"]
    kt, kx, kx0, ky0 = jax.random.split(key, 4)
    ts = jax.random.uniform(kt, (N_times,), minval=1e-12, maxval=t_max_phys)
    xs = jax.random.uniform(kx, (N_space,), minval=x0b, maxval=x1b)
    ys = jnp.full((N_space,), y0b)
    x0s = jax.random.uniform(kx0, (N_space,), minval=cfg_xy0["x_bounds"][0], maxval=cfg_xy0["x_bounds"][1])
    y0s = jax.random.uniform(ky0, (N_space,), minval=cfg_xy0["y_bounds"][0], maxval=cfg_xy0["y_bounds"][1])

    def pack(t):
        T  = jnp.full((N_space,), t)
        tN, xN, yN, x0N, y0N = normalize_txyx0y0(T, xs, ys, x0s, y0s, cfg_norm_xy, cfg_xy0)
        return jnp.stack([tN, xN, yN, x0N, y0N], axis=1)
    return jax.vmap(pack)(ts)

def sample_outside_ring_param(cfg_orig_xy: Dict, cfg_exp_xy: Dict, cfg_xy0: Dict,
                              N_space: int, N_times: int, *,
                              key: jax.Array, t_max_phys: float):
    (x0e,x1e),(y0e,y1e) = cfg_exp_xy["x_bounds"],  cfg_exp_xy["y_bounds"]
    (x0o,x1o),(y0o,y1o) = cfg_orig_xy["x_bounds"], cfg_orig_xy["y_bounds"]
    kt,kx,ky,kx0,ky0 = jax.random.split(key,5)
    ts  = jax.random.uniform(kt,  (N_times,), minval=0.0, maxval=t_max_phys)
    xs  = jax.random.uniform(kx,  (N_space,), minval=x0e, maxval=x1e)
    ys  = jax.random.uniform(ky,  (N_space,), minval=y0e, maxval=y1e)
    x0s = jax.random.uniform(kx0, (N_space,), minval=cfg_xy0["x_bounds"][0], maxval=cfg_xy0["x_bounds"][1])
    y0s = jax.random.uniform(ky0, (N_space,), minval=cfg_xy0["y_bounds"][0], maxval=cfg_xy0["y_bounds"][1])

    mask_out = (xs < x0o) | (xs > x1o) | (ys < y0o) | (ys > y1o)
    def pack(t):
        T  = jnp.full((N_space,), t)
        tN, xN, yN, x0N, y0N = normalize_txyx0y0(T, xs, ys, x0s, y0s, cfg_exp_xy, cfg_xy0)
        return jnp.stack([tN[mask_out], xN[mask_out], yN[mask_out], x0N[mask_out], y0N[mask_out]], axis=1)
    return jax.vmap(pack)(ts)

def sample_mass_batches_param(cfg_xy: Dict, cfg_xy0: Dict,
                              N_space: int, N_times: int, *,
                              key: jax.Array, t_max_phys: float, include_t0: bool = True):
    (x0,x1),(y0,y1) = cfg_xy["x_bounds"], cfg_xy["y_bounds"]
    if include_t0:
        N_times = max(1, N_times)
        t_list = [0.0]
        if N_times > 1:
            kt = jax.random.split(key, 1)[0]
            ts_rand = jax.random.uniform(kt, (N_times-1,), minval=1e-12, maxval=t_max_phys)
            t_list += list(ts_rand)
        times = jnp.array(t_list)
    else:
        kt = jax.random.split(key, 1)[0]
        times = jax.random.uniform(kt, (N_times,), minval=1e-12, maxval=t_max_phys)

    kx, ky, kx0, ky0 = jax.random.split(key, 4)
    xs  = jax.random.uniform(kx,  (N_space,), minval=x0, maxval=x1)
    ys  = jax.random.uniform(ky,  (N_space,), minval=y0, maxval=y1)
    x0s = jax.random.uniform(kx0, (N_space,), minval=cfg_xy0["x_bounds"][0], maxval=cfg_xy0["x_bounds"][1])
    y0s = jax.random.uniform(ky0, (N_space,), minval=cfg_xy0["y_bounds"][0], maxval=cfg_xy0["y_bounds"][1])

    def pack(t):
        T  = jnp.full((N_space,), t)
        tN, xN, yN, x0N, y0N = normalize_txyx0y0(T, xs, ys, x0s, y0s, cfg_xy, cfg_xy0)
        return jnp.stack([tN, xN, yN, x0N, y0N], axis=1)
    batches = jax.vmap(pack)(times)
    return batches, times

# ==========================
# 3) PINN 5D model
# ==========================

class PINN_TXYX0Y0(eqx.Module):
    mlp: eqx.nn.MLP
    bias0: float = eqx.field(static=True)
    def __init__(self, *, width: int = 256, depth: int = 6, key: jax.Array, bias0: float = 6.5):
        self.mlp = eqx.nn.MLP(in_size=5, out_size=1, width_size=width, depth=depth,
                              activation=jax.nn.tanh, key=key)
        self.bias0 = bias0
    def __call__(self, txyx0y0N: jnp.ndarray) -> jnp.ndarray:
        z = self.mlp(txyx0y0N)[..., 0] - self.bias0
        s = jax.nn.softplus(z)    # ≥0
        out = 1.0 - jnp.exp(-s)   # ∈ (0,1]
        return out

# ==========================
# 4) PDE and losses
# ==========================

def _scales(cfg: Dict) -> Tuple[float, float, float]:
    tmax = float(cfg["t_bounds"][1])
    (x0,x1),(y0,y1) = cfg["x_bounds"], cfg["y_bounds"]
    Lx, Ly = float(x1-x0), float(y1-y0)
    return tmax, Lx, Ly

def pde_residual_advective_5D(model, cfg: Dict, txyx0y0N: jnp.ndarray) -> jnp.ndarray:
    tmax, Lx, Ly = _scales(cfg)
    De  = float(cfg.get("D_e", 0.0))
    vx  = float(cfg.get("vx", 0.0))
    vy  = float(cfg.get("vy", 0.0))
    tau = float(cfg.get("tau_e", 0.0))

    def fN(z): return model(z)

    def residual_one(z):
        val = fN(z)
        g   = jax.jacfwd(fN)(z)          # grad w.r.t. (tN,xN,yN,x0N,y0N)
        dtN, dxN, dyN = g[0], g[1], g[2]
        if De > 0.0:
            H = jax.hessian(fN)(z)
            dxxN, dyyN = H[1,1], H[2,2]
        else:
            dxxN = dyyN = 0.0

        ft  = dtN / tmax
        fx  = dxN / Lx
        fy  = dyN / Ly
        fxx = dxxN / (Lx*Lx)
        fyy = dyyN / (Ly*Ly)

        term_adv  = vx*fx + vy*fy
        term_diff = -De*(fxx + fyy)
        term_trap = (val / tau) if tau > 0.0 else 0.0
        return ft + term_adv + term_diff + term_trap

    return jax.vmap(residual_one)(txyx0y0N)

def _pred_batch(model, txyx0y0N: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda z: model(z))(txyx0y0N)

def loss_ic_weighted_5D(params, static_model, batch_ic_5d, xs_phys, ys_phys, x0s_phys, y0s_phys,
                        sigma: float, alpha=0.9):
    model = eqx.combine(params, static_model)
    s2 = sigma*sigma
    target = jnp.exp(-(((xs_phys - x0s_phys)**2 + (ys_phys - y0s_phys)**2) / (2.0*s2)))
    pred   = _pred_batch(model, batch_ic_5d)
    w_raw  = target * target
    w      = w_raw / (jnp.mean(w_raw) + 1e-16)
    w      = (1.0 - alpha) + alpha * w
    return jnp.mean(w * (pred - target) ** 2)

def make_anchor_sets_5D(cfg_orig_xy, cfg_norm_xy, cfg_xy0, *, vy: float, times, N: int, key, sigma: float):
    (x0,x1),(y0,y1) = cfg_orig_xy["x_bounds"], cfg_orig_xy["y_bounds"]
    kx, ky, kx0, ky0 = jax.random.split(key, 4)
    xs  = jax.random.uniform(kx,  (N,), minval=x0, maxval=x1)
    ys  = jax.random.uniform(ky,  (N,), minval=y0, maxval=y1)
    x0s = jax.random.uniform(kx0, (N,), minval=x0, maxval=x1)
    y0s = jax.random.uniform(ky0, (N,), minval=y0, maxval=y1)
    s2  = sigma*sigma

    def pack_for_t(t):
        T = jnp.full((N,), t)
        y_back = ys - vy * t
        targ = jnp.exp(-(((xs - x0s)**2 + (y_back - y0s)**2) / (2.0*s2)))
        tN, xN, yN, x0N, y0N = normalize_txyx0y0(T, xs, ys, x0s, y0s, cfg_norm_xy, cfg_xy0)
        txyx0y0 = jnp.stack([tN, xN, yN, x0N, y0N], axis=1)
        return txyx0y0, targ

    txy_list, tgt_list = zip(*[pack_for_t(float(ti)) for ti in times])
    return jnp.stack(txy_list,0), jnp.stack(tgt_list,0)

def _make_loss_pde_adv_5D(cfg: dict):
    def loss_pde(params, static_model, txyx0y0_pde: jnp.ndarray) -> jnp.ndarray:
        model = eqx.combine(params, static_model)
        R = pde_residual_advective_5D(model, cfg, txyx0y0_pde)
        return jnp.mean(R * R)
    return loss_pde

def loss_anchor_5D(params, static_model, txyx0y0_anchor, tgt_anchor):
    model = eqx.combine(params, static_model)
    pred_t = jax.vmap(lambda arr: _pred_batch(model, arr))(txyx0y0_anchor)
    return jnp.mean((pred_t - tgt_anchor) ** 2)

def loss_zero_on_batches_5D(params, static_model, batches_5d):
    model = eqx.combine(params, static_model)
    def at_time(batch):
        vals = _pred_batch(model, batch)
        return jnp.mean(vals * vals)
    Ls = jax.vmap(at_time)(batches_5d)
    return jnp.mean(Ls)

def loss_mass_5D(params, static_model, mass_batches_5d: jnp.ndarray,
                 mass_times: jnp.ndarray, area: float, tau_e: float) -> jnp.ndarray:
    if tau_e <= 0.0:
        return jnp.array(0.0, dtype=jnp.float32)
    model = eqx.combine(params, static_model)
    def Q_from_batch(batch):
        vals = _pred_batch(model, batch)
        return area * jnp.mean(vals)
    Qs = jax.vmap(Q_from_batch)(mass_batches_5d)
    Q0 = Qs[0]
    target = Q0 * jnp.exp(-mass_times / tau_e)
    rel_err = (Qs - target) / (Q0 + 1e-16)
    return jnp.mean(rel_err ** 2)

# ==========================
# 5) Training with early stopping
# ==========================

@eqx.filter_jit
def _step(params, static_model, opt_state,
          batch_ic_5d, xs_ic_b, ys_ic_b, x0s_ic_b, y0s_ic_b,
          batch_pde_5d, bc_batches_5d, ring_batches_5d,
          txyx0y0_anchor, tgt_anchor,
          mass_batches_5d, mass_times, area_exp, tau_e,
          weights, scales, optimizer,
          sigma_ic, loss_pde_fn, alpha_ic,
          lam_anchor, lam_bc, lam_ring, w_mass):
    w_ic, w_pde = weights
    s_ic, s_pde = scales
    eps = 1e-12

    def total_loss(p):
        lic  = loss_ic_weighted_5D(p, static_model, batch_ic_5d, xs_ic_b, ys_ic_b, x0s_ic_b, y0s_ic_b,
                                   sigma=sigma_ic, alpha=alpha_ic)
        lpde = loss_pde_fn(p, static_model, batch_pde_5d)
        lanch = loss_anchor_5D(p, static_model, txyx0y0_anchor, tgt_anchor)
        lbc   = loss_zero_on_batches_5D(p, static_model, bc_batches_5d)
        lring = loss_zero_on_batches_5D(p, static_model, ring_batches_5d)
        lmass = loss_mass_5D(p, static_model, mass_batches_5d, mass_times, area_exp, tau_e)

        lic_n  = lic  / (s_ic + eps)
        lpde_n = lpde / (s_pde + eps)
        L = (w_ic*lic_n + w_pde*lpde_n) + w_mass*lmass + lam_anchor*lanch + lam_bc*lbc + lam_ring*lring
        return L, (lic, lpde, lmass, lanch, lbc, lring)

    (value, (lic, lpde, lmass, lanch, lbc, lring)), grads = eqx.filter_value_and_grad(total_loss, has_aux=True)(params)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value, lic, lpde, lmass, lanch, lbc, lring

def train_curriculum_5D(model, cfg_orig: dict, cfg_exp: dict, *,
                        key, epochs=1200,
                        N_ic=65536, N_pde=131072,
                        N_bc_space=16384, N_bc_times=8,
                        N_ring_space=32768, N_ring_times=6,
                        N_anchor=8192, anchor_times=None,
                        N_mass_space=8192, N_mass_times=6, lam_mass=0.3,
                        lr=3e-4, batch=16384,
                        lam_ic=6.0, lam_pde=1.0, lam_anchor=4.0, lam_bc=0.0, lam_ring=6.0,
                        ramp_epochs=800, ema_beta=0.95, t_cross: float = None, log_every=100,
                        sigma_ic=9e-5,
                        early_stopping: bool = True, patience: int = 120, min_delta: float = 0.0):
    optimizer = optax.adamw(lr)
    params, static_model = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)

    vy = float(cfg_orig["vy"]); y0, y1 = cfg_orig["y_bounds"]
    t_cross_max = (y1 - y0) / abs(vy) if abs(vy) > 0 else cfg_orig["sim_time"]
    horizon = t_cross if t_cross is not None else t_cross_max
    t_phys_max = float(min(cfg_orig["sim_time"], 1.3 * horizon))
    loss_pde_fn = _make_loss_pde_adv_5D(cfg_exp)

    logs = dict(total=[], ic=[], pde=[], mass=[], anchor=[], bc=[], ring=[], pde_rms=[])

    # Anchors
    key, kanchor = jax.random.split(key, 2)
    if anchor_times is None:
        anchor_times = [0.1*t_phys_max, 0.3*t_phys_max, 0.6*t_phys_max, 0.9*t_phys_max, 1.2*t_phys_max]
    txyx0y0_anchor, tgt_anchor = make_anchor_sets_5D(cfg_orig, cfg_exp, cfg_orig,
                                                     vy=float(cfg_orig["vy"]),
                                                     times=anchor_times, N=N_anchor, key=kanchor,
                                                     sigma=sigma_ic)

    # Mass
    tau_e = float(cfg_exp.get("tau_e", 0.0))
    (x0e,x1e),(y0e,y1e) = cfg_exp["x_bounds"], cfg_exp["y_bounds"]
    area_exp = float((x1e - x0e) * (y1e - y0e))
    key, kmass = jax.random.split(key, 2)
    mass_batches_5d, mass_times = sample_mass_batches_param(cfg_exp, cfg_orig,
                                                            N_mass_space, N_mass_times,
                                                            key=kmass, t_max_phys=t_phys_max, include_t0=True)
    s_ic = jnp.array(1.0); s_pde = jnp.array(1.0)

    print("[CHECK] 5D model: input=5 ⇒", getattr(static_model, "mlp").in_size if hasattr(static_model, "mlp") else "unknown")

    best_metric = float("inf")   # pde_RMS
    best_epoch = 0
    best_params = params
    epochs_no_improve = 0

    for ep in range(1, epochs+1):
        key, k_ic, k_pde, k_bc, k_ring, k_perm = jax.random.split(key, 6)

        batch_ic_5d, xs_ic, ys_ic, x0s_ic, y0s_ic, _ = sample_ic_points_param(cfg_orig, cfg_exp, cfg_orig,
                                                                              N_ic, key=k_ic, sigma=sigma_ic)
        txyx0y0_pde_all = sample_pde_points_param(cfg_exp, cfg_orig, N_pde, key=k_pde, t_max_phys=t_phys_max)
        perm = jax.random.permutation(k_perm, txyx0y0_pde_all.shape[0])
        txyx0y0_pde_all = txyx0y0_pde_all[perm]

        bc_batches_5d   = sample_bc_anode_points_param(cfg_orig, cfg_exp, cfg_orig, N_bc_space, N_bc_times, key=k_bc, t_max_phys=t_phys_max)
        ring_batches_5d = sample_outside_ring_param(cfg_orig, cfg_exp, cfg_orig, N_ring_space, N_ring_times, key=k_ring, t_max_phys=t_phys_max)

        if ep == 1:
            w_ic, w_pde, w_mass = lam_ic, 0.0, 0.0
        else:
            alpha = min(1.0, max(0.0, (ep-1)/float(max(ramp_epochs,1))))
            w_ic, w_pde = lam_ic, lam_pde*alpha
            w_mass = lam_mass*alpha if tau_e > 0.0 else 0.0

        nb = int(np.ceil(txyx0y0_pde_all.shape[0] / batch))
        tot=icv=pdev=massv=anchv=bcv=ringv=0.0; seen=0

        def ema(s, v): return ema_beta*s + (1.0-ema_beta)*max(v,1e-16)

        for i in range(nb):
            s,e = i*batch, min((i+1)*batch, txyx0y0_pde_all.shape[0])
            idx_ic = jnp.mod(jnp.arange(s, e), batch_ic_5d.shape[0])
            batch_ic_b = batch_ic_5d[idx_ic]
            xs_b, ys_b   = xs_ic[idx_ic],  ys_ic[idx_ic]
            x0s_b, y0s_b = x0s_ic[idx_ic], y0s_ic[idx_ic]
            batch_pde_b  = txyx0y0_pde_all[s:e]

            params, opt_state, Ltot, Lic, Lpde, Lmass, Lanch, Lbc, Lring = _step(
                params, static_model, opt_state,
                batch_ic_b, xs_b, ys_b, x0s_b, y0s_b,
                batch_pde_b, bc_batches_5d, ring_batches_5d,
                txyx0y0_anchor, tgt_anchor,
                mass_batches_5d, mass_times, area_exp, tau_e,
                (w_ic, w_pde), (s_ic, s_pde), optimizer,
                sigma_ic, loss_pde_fn, 0.9,
                lam_anchor, lam_bc, lam_ring, w_mass
            )

            m = int(e-s)
            tot += float(Ltot)*m; icv += float(Lic)*m; pdev += float(Lpde)*m
            massv += float(Lmass)*m; anchv += float(Lanch)*m
            bcv += float(Lbc)*m; ringv += float(Lring)*m; seen += m

        L_ic_ep, L_pde_ep, L_mass_ep = icv/seen, pdev/seen, massv/seen
        logs["total"].append(tot/seen); logs["ic"].append(L_ic_ep)
        logs["pde"].append(L_pde_ep);  logs["mass"].append(L_mass_ep)
        logs["anchor"].append(anchv/seen); logs["bc"].append(bcv/seen); logs["ring"].append(ringv/seen)
        s_ic, s_pde = ema(s_ic, L_ic_ep), ema(s_pde, L_pde_ep)

        kk = jax.random.split(jax.random.PRNGKey(ep), 1)[0]
        txyx0y0_diag = sample_pde_points_param(cfg_exp, cfg_orig, 4096, key=kk, t_max_phys=t_phys_max)
        R = pde_residual_advective_5D(eqx.combine(params, static_model), cfg_exp, txyx0y0_diag)
        pde_rms = float(jnp.sqrt(jnp.mean(R*R)))
        logs["pde_rms"].append(pde_rms)

        # Early stopping based on pde_RMS
        if pde_rms < best_metric - float(min_delta):
            best_metric = pde_rms
            best_epoch = ep
            best_params = params
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(f"[EARLY-STOP] ep={ep:4d} without improvement in pde_RMS for {patience} epochs "
                      f"(best={best_metric:.3e} at ep={best_epoch}).")
                break

        if ep == 1 or (ep % log_every) == 0:
            print(f"[PINN-5D] ep {ep:4d} | L={logs['total'][-1]:.3e} | "
                  f"LIC={logs['ic'][-1]:.3e} | LPDE={logs['pde'][-1]:.3e} | LMASS={logs['mass'][-1]:.3e} | "
                  f"Lanch={logs['anchor'][-1]:.3e} | Lbc={logs['bc'][-1]:.3e} | Lring={logs['ring'][-1]:.3e} | "
                  f"pde_RMS={logs['pde_rms'][-1]:.3e}")

    # If we use early stopping (Phase A), we keep the best model
    # according to pde_RMS. If we do NOT use early stopping (Phase B/C),
    # we use the last epoch.
    if early_stopping:
        params_to_use = best_params
        logs["best_epoch"] = best_epoch
    else:
        params_to_use = params
        logs["best_epoch"] = epochs

    logs["best_pde_rms"] = float(best_metric)   # still store the best pde_RMS

    model_tr = eqx.combine(params_to_use, static_model)
    return model_tr, logs, t_phys_max

# ==========================
# 6) Visualization helpers
# ==========================

NORMALIZE_VIS = True  # normalize each frame to its local maximum

def plot_map_01(Z, X=None, Y=None, title: str = "",
                cmap: str = "coolwarm", normalize_max: bool = False,
                mask=None):
    Z = np.asarray(np.clip(Z, 0.0, 1.0))
    if normalize_max and Z.max() > 0:
        Z = Z / float(Z.max())

    if mask is not None:
        mask_np = np.asarray(mask, dtype=bool)
        Z = np.ma.array(Z, mask=~mask_np)
        cmap_obj = matplotlib.cm.get_cmap(cmap).with_extremes(bad="0.7")
    else:
        cmap_obj = matplotlib.cm.get_cmap(cmap)

    fig, ax = plt.subplots()
    if X is None or Y is None:
        im = ax.imshow(Z.T, origin="lower", aspect="equal",
                       vmin=0.0, vmax=1.0, cmap=cmap_obj, interpolation="nearest")
    else:
        Xn = np.asarray(X); Yn = np.asarray(Y)
        extent = [Xn.min(), Xn.max(), Yn.min(), Yn.max()]
        im = ax.imshow(Z.T, origin="lower", extent=extent,
                       aspect="equal", vmin=0.0, vmax=1.0,
                       cmap=cmap_obj, interpolation="nearest")
        ax.set_aspect("equal", adjustable="box")

    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_map_electron_moving(
    Z,
    X,
    Y,
    cfg_orig: dict,
    title: str = "",
    normalize_max: bool = False,
):
    """
    2D map for the temporal evolution of the charge packet.
    - Divergent colormap (coolwarm).
    - Region outside the original box in gray.
    - Same scale in x and y (perfect circle in all figures).
    """
    Z = np.asarray(np.clip(Z, 0.0, 1.0))
    if normalize_max and Z.max() > 0:
        Z = Z / float(Z.max())

    Xn = np.asarray(X)
    Yn = np.asarray(Y)
    extent = [Xn.min(), Xn.max(), Yn.min(), Yn.max()]

    (x0o, x1o), (y0o, y1o) = cfg_orig["x_bounds"], cfg_orig["y_bounds"]
    mask_inside = (Xn >= x0o) & (Xn <= x1o) & (Yn >= y0o) & (Yn <= y1o)

    # Transpose to be consistent with imshow
    Z_plot = Z.T
    mask_plot = (~mask_inside).T  # True = outside ⇒ gray

    Z_masked = np.ma.array(Z_plot, mask=mask_plot)

    cmap = plt.get_cmap("coolwarm").copy()  # divergent
    cmap.set_bad(color="0.5")  # gray for outside the original box

    fig, ax = plt.subplots()
    im = ax.imshow(
        Z_masked,
        origin="lower",
        extent=extent,
        aspect="equal",          # <- same scale in x and y
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax

def draw_original_box(ax, bounds_original: Tuple[Tuple[float,float], Tuple[float,float]]):
    (x0,x1), (y0,y1) = bounds_original
    ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0], color="white", linewidth=1.5)

def plot_curve(values, title="Loss", xlabel="Epoch", ylabel="MSE"):
    fig, ax = plt.subplots()
    ax.plot(np.asarray(values))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3); fig.tight_layout()
    return fig, ax

def center_of_mass(Z: np.ndarray, X: np.ndarray, Y: np.ndarray, mask=None):
    Zp = np.clip(Z, 0.0, 1.0)
    if mask is not None:
        Zp = Zp * mask
    s = Zp.sum()
    if s < 1e-16:
        return np.nan, np.nan
    return float((Zp*X).sum()/s), float((Zp*Y).sum()/s)

def widths_xy(Z: np.ndarray, X: np.ndarray, Y: np.ndarray, mask=None):
    """Widths σx, σy (in meters) as standard deviations of the packet inside the box."""
    Zp = np.clip(Z, 0.0, 1.0)
    if mask is not None:
        Zp = Zp * mask
    s = Zp.sum()
    if s < 1e-16:
        return np.nan, np.nan
    xc = (Zp * X).sum() / s
    yc = (Zp * Y).sum() / s
    sx = np.sqrt((Zp * (X - xc) ** 2).sum() / s)
    sy = np.sqrt((Zp * (Y - yc) ** 2).sum() / s)
    return float(sx), float(sy)

def make_grid_at_time_for_seed(cfg_xy: dict, t_phys: float, x0_seed: float, y0_seed: float, cfg_xy0: dict):
    nx, ny = int(cfg_xy["nx"]), int(cfg_xy["ny"])
    x0, x1 = cfg_xy["x_bounds"]; y0, y1 = cfg_xy["y_bounds"]
    xs = jnp.linspace(x0, x1, nx, endpoint=False)
    ys = jnp.linspace(y0, y1, ny, endpoint=False)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy"); Xg, Yg = X.T, Y.T
    Tg = jnp.full_like(Xg, t_phys)
    X0g = jnp.full_like(Xg, x0_seed); Y0g = jnp.full_like(Yg, y0_seed)
    tN, xN, yN, x0N, y0N = normalize_txyx0y0(Tg, Xg, Yg, X0g, Y0g, cfg_xy, cfg_xy0)
    T5 = jnp.stack([tN.ravel(), xN.ravel(), yN.ravel(), x0N.ravel(), y0N.ravel()], axis=1)
    return T5, Xg, Yg

def evaluate_on_grid_5D_seed(model, T5: jnp.ndarray, nx: int, ny: int, batch=8192):
    outs = []
    f_one = lambda z: model(z)
    for s in range(0, T5.shape[0], batch):
        e = min(T5.shape[0], s + batch)
        outs.append(jax.vmap(f_one)(T5[s:e]))
    Z = jnp.concatenate(outs, axis=0).reshape(nx, ny)
    return jnp.clip(Z, 0.0, 1.0)

def total_mass_on_grid_seed(model, cfg_xy: dict, cfg_xy0: dict, t_phys: float,
                            x0_seed: float, y0_seed: float, batch=8192) -> float:
    nx, ny = int(cfg_xy["nx"]), int(cfg_xy["ny"])
    (x0,x1),(y0,y1) = cfg_xy["x_bounds"], cfg_xy["y_bounds"]
    area = float((x1-x0)*(y1-y0))
    T5, Xg, Yg = make_grid_at_time_for_seed(cfg_xy, t_phys, x0_seed, y0_seed, cfg_xy0)
    Zt = np.asarray(evaluate_on_grid_5D_seed(model, T5, nx, ny, batch=batch))
    return area * float(np.mean(Zt))

def plot_com_trajectory(times, com_y, x0_seed, y0_seed):
    fig, ax = plt.subplots()
    ax.plot(times, com_y, marker="o")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("COM_y (m)")
    ax.set_title(f"Center of mass trajectory (y)  seed=({x0_seed:.2e},{y0_seed:.2e})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

def plot_width_evolution(times, sigma_x, sigma_y, x0_seed, y0_seed):
    fig, ax = plt.subplots()
    ax.plot(times, sigma_x, marker="o", label="σ_x(t)")
    ax.plot(times, sigma_y, marker="s", label="σ_y(t)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Width (m)")
    ax.set_title(f"Packet broadening by diffusion  seed=({x0_seed:.2e},{y0_seed:.2e})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

# === Panel-type and “circles” figures (we can still use them for extra plots) ===

def plot_mass_evolution_panels_for_seed(X, Y, Z_list, times, mask_orig,
                                        bounds_original, y_anode, x0_seed, y0_seed):
    """
    Panel of subplots with the full evolution of the mass f(t,x,y;x0,y0)
    for one seed (each column is one time instant).
    """
    Z_list = [np.asarray(np.clip(Z, 0.0, 1.0)) for Z in Z_list]
    times = list(times)
    n_frames = len(Z_list)
    n_cols = min(4, n_frames)
    n_rows = int(np.ceil(n_frames / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    Xn = np.asarray(X); Yn = np.asarray(Y)
    extent = [Xn.min(), Xn.max(), Yn.min(), Yn.max()]
    cmap_obj = matplotlib.cm.get_cmap("coolwarm").with_extremes(bad="0.7")

    mask_np = np.asarray(mask_orig, dtype=bool)

    for idx, (Zt, tt) in enumerate(zip(Z_list, times)):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        Zmasked = np.ma.array(Zt, mask=~mask_np)
        im = ax.imshow(
            Zmasked.T,
            origin="lower",
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap_obj,
            interpolation="nearest"
        )
        ax.axhline(y=y_anode, color="white", linestyle="--", linewidth=1.0)
        draw_original_box(ax, bounds_original)
        ax.set_title(f"t={tt:.2e} s")
        ax.set_aspect("equal", adjustable="box")

    # Hide empty subplots if any
    for idx in range(n_frames, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")

    fig.suptitle(f"Mass evolution  seed=({x0_seed:.2e},{y0_seed:.2e})")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

def plot_diffusion_circles_for_seed(X, Y, times, com_x, com_y, sigma_x, sigma_y,
                                    bounds_original, y_anode, x0_seed, y0_seed):
    """
    “Circles” figure that grows:
      - Each circle is centered at the center of mass (COM_x, COM_y)
      - The radius is proportional to the width (σ_x, σ_y) at that instant.
    """
    Xn = np.asarray(X); Yn = np.asarray(Y)
    (x0o,x1o), (y0o,y1o) = bounds_original

    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    ax.set_xlim(x0o, x1o)
    ax.set_ylim(y0o, y1o)
    ax.set_aspect("equal", adjustable="box")

    # Original detector rectangle
    draw_original_box(ax, bounds_original)
    # Anode line
    ax.axhline(y=y_anode, color="black", linestyle="--", linewidth=1.0)

    times = np.asarray(times, dtype=float)
    com_x = np.asarray(com_x, dtype=float)
    com_y = np.asarray(com_y, dtype=float)
    sigma_x = np.asarray(sigma_x, dtype=float)
    sigma_y = np.asarray(sigma_y, dtype=float)

    n = len(times)
    cmap = matplotlib.cm.get_cmap("viridis")

    for i in range(n):
        if not np.isfinite(com_x[i]) or not np.isfinite(com_y[i]):
            continue
        if not np.isfinite(sigma_x[i]) or not np.isfinite(sigma_y[i]):
            continue
        # Average radius from σx, σy
        r = max( (sigma_x[i] + sigma_y[i]) / 2.0, 1e-8 )
        color = cmap(i / max(n-1, 1))
        circ = patches.Circle(
            (com_x[i], com_y[i]),
            radius=r,
            fill=False,
            linestyle="-",
            linewidth=1.5,
            edgecolor=color,
            alpha=0.8,
            label=None
        )
        ax.add_patch(circ)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Diffusion growth (circles)  seed=({x0_seed:.2e},{y0_seed:.2e})")
    fig.tight_layout()
    return fig

def plot_superposed_mass_trajectory(
    Z_snapshots,
    X,
    Y,
    cfg_orig: dict,
    bounds_original,
    model_label: str,
    x0_seed: float,
    y0_seed: float,
    normalize_vis: bool = True,
):
    """
    A single map with the sum of all snapshots Z_snapshots:
    “mass being dragged” (all times superposed).
    """
    if not Z_snapshots:
        return None, None

    Z_traj = np.zeros_like(np.asarray(Z_snapshots[0]))
    for Zt in Z_snapshots:
        Z_traj += np.clip(np.asarray(Zt), 0.0, 1.0)

    max_val = float(Z_traj.max())
    if normalize_vis and max_val > 0.0:
        Z_traj = Z_traj / max_val

    title = (
        f"[{model_label}] Trajectory (all times superposed)  "
        f"seed=({x0_seed:.2e},{y0_seed:.2e})"
    )

    fig_traj, ax_traj = plot_map_electron_moving(
        Z_traj,
        X,
        Y,
        cfg_orig,
        title=title,
        normalize_max=False,  # already normalized
    )
    draw_original_box(ax_traj, bounds_original)
    return fig_traj, ax_traj

def plot_com_path_with_diffusion(
    com_x,
    com_y,
    sigma_x,
    sigma_y,
    time_snapshots,
    bounds_original,
    model_label: str,
    x0_seed: float,
    y0_seed: float,
    y_anode: float,
):
    """
    COM path in the (x,y) plane + diffusion circles:
      - Line joining center-of-mass positions.
      - One circle at each instant with radius ~ (σx+σy)/2.
      - Horizontal anode line.
    """
    com_x = np.asarray(com_x, dtype=float)
    com_y = np.asarray(com_y, dtype=float)
    sigma_x = np.asarray(sigma_x, dtype=float)
    sigma_y = np.asarray(sigma_y, dtype=float)
    time_snapshots = np.asarray(time_snapshots, dtype=float)

    (x0o, x1o), (y0o, y1o) = bounds_original

    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    ax.set_xlim(x0o, x1o)
    ax.set_ylim(y0o, y1o)
    ax.set_aspect("equal", adjustable="box")

    draw_original_box(ax, bounds_original)
    ax.axhline(y=y_anode, linestyle="--", linewidth=1.0)

    # COM path
    ax.plot(com_x, com_y, linewidth=1.0, marker="o", label="COM path")

    n_snap = len(time_snapshots)
    if n_snap > 0:
        colors = plt.cm.viridis(np.linspace(0.0, 1.0, n_snap))
    else:
        colors = []

    for i, (cx, cy, sx, sy, col) in enumerate(
        zip(com_x, com_y, sigma_x, sigma_y, colors)
    ):
        if not np.isfinite(cx) or not np.isfinite(cy):
            continue
        if not np.isfinite(sx) or not np.isfinite(sy):
            continue
        r = max((sx + sy) / 2.0, 1e-10)
        circ = patches.Circle(
            (cx, cy),
            radius=r,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            edgecolor=col,
            alpha=0.8,
        )
        ax.add_patch(circ)
        ax.text(
            cx,
            cy,
            f"t{i}",
            fontsize=6,
            ha="center",
            va="center",
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"[{model_label}] COM path + diffusion circles  "
        f"seed=({x0_seed:.2e},{y0_seed:.2e})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax

# ==========================
# 7) Global hyperparameters
# ==========================

EPOCHS = 600
BATCH  = 16_384
N_IC, N_PDE = 65_536, 131_072
N_BC_SPACE, N_BC_TIMES = 16_384, 8
N_RING_SPACE, N_RING_TIMES = 32_768, 6
N_MASS_SPACE, N_MASS_TIMES = 8_192, 6
N_ANCHOR = 8_192

LAM_MASS_BASE   = 0.30
LAM_IC_BASE     = 6.0
LAM_PDE_BASE    = 1.0
LAM_ANCHOR_BASE = 4.0
LAM_BC_BASE     = 1.0
LAM_RING_BASE   = 6.0

WIDTH, DEPTH = 256, 6
RAMP = 600          # full ramp exactly at epoch 600
LR = 3e-4
LOG_EVERY = 100

EARLY_STOPPING   = True
EARLY_PATIENCE   = 120
EARLY_MIN_DELTA  = 0.0

N0_SIGMA = 9e-5

# Standard configuration labels
FULL_LABEL = "MASS+ANCHOR+BC+RING"
BEST_ABLATION_LABEL = "MASS+BC+RING"

# Evaluation seeds (initial positions)
seeds_eval = [
    (1.5e-3, 2.5e-3),
    (0.6e-3, 4.4e-3),
    (2.4e-3, 3.6e-3),
    (1.0e-3, 1.2e-3),
]

# ==========================
# 8) Full diagnostics for the FULL model
# ==========================

def times_for_seed(y0_seed: float, sim_time: float, vy: float, y_anode: float):
    if abs(vy) < 1e-30:
        base = sim_time * 0.5
    else:
        base = max((y0_seed - y_anode)/abs(vy), 1e-12)
    cand = np.array([0.0, 0.25*base, 0.5*base, 1.0*base, 1.25*base], dtype=float)
    cand = np.clip(cand, 0.0, float(sim_time))
    if base <= sim_time and all(abs(t-base) > 1e-15 for t in cand):
        cand = np.sort(np.append(cand, base))
    cand = np.unique(np.round(cand, 12))
    return cand.tolist()

def run_full_model_diagnostics(
    model_tr,
    cfg_orig: dict,
    cfg_exp: dict,
    t_phys_max: float,
    logs: Optional[dict] = None,
    model_label: str = "FULL",
):
    """
    Full diagnostics for a trained model (by default FULL):
      - Checks at t=0 against theoretical Gaussian.
      - Maps f(t,x,y;x0,y0) over time (divergent colormap, gray outside).
      - Panel plots of mass evolution (all epochs) for the 4 seeds.
      - Overlaid contours of packet trajectory.
      - "Circles" diffusion figures for the 4 seeds.
      - COM_y(t) and σ_x(t), σ_y(t) trajectories for each seed.
      - Training loss curves (total, IC, PDE, MASS, ANCHOR, BC, RING, PDE_RMS).
      - Mass decay vs theoretical exponential (seed 0) + extra COM/σ for seed 0.
      - Map with all times superposed + COM path + diffusion circles.
    """
    print(f"[EVAL {model_label}] Seeds (x0,y0):", seeds_eval)

    nxE, nyE = int(cfg_exp["nx"]), int(cfg_exp["ny"])
    (x0o, x1o), (y0o, y1o) = cfg_orig["x_bounds"], cfg_orig["y_bounds"]
    bounds_original = cfg_exp["bounds_original"]
    mask_info_printed = False
    y_anode = y0o
    vy = float(cfg_orig["vy"])

    for (x0_seed, y0_seed) in seeds_eval:
        # ---- t = 0: comparison with theoretical Gaussian ----
        T5_0, X0, Y0 = make_grid_at_time_for_seed(
            cfg_exp,
            t_phys=0.0,
            x0_seed=x0_seed,
            y0_seed=y0_seed,
            cfg_xy0=cfg_orig,
        )
        Z0 = np.asarray(
            evaluate_on_grid_5D_seed(
                model_tr, T5_0, nxE, nyE, batch=min(BATCH, 8192)
            )
        )
        N0 = np.exp(
            -(
                ((np.asarray(X0) - x0_seed) ** 2
                 + (np.asarray(Y0) - y0_seed) ** 2)
                / (2.0 * N0_SIGMA * N0_SIGMA)
            )
        )

        mask_orig = (X0 >= x0o) & (X0 <= x1o) & (Y0 >= y0o) & (Y0 <= y1o)
        A_orig = (x1o - x0o) * (y1o - y0o)
        mass_pred0 = A_orig * float((np.clip(Z0, 0, 1) * mask_orig).mean())
        mass_tgt0 = A_orig * float((np.clip(N0, 0, 1) * mask_orig).mean())

        def _com(Z):
            return center_of_mass(np.clip(Z * mask_orig, 0, 1), X0, Y0)

        xc_p, yc_p = _com(Z0)
        xc_t, yc_t = _com(np.clip(N0, 0, 1))
        mse0 = float(np.mean((Z0 - np.clip(N0, 0, 1)) ** 2))

        print(
            f"[{model_label} CHECK@t0 seed=({x0_seed:.2e},{y0_seed:.2e})] "
            f"MSE={mse0:.3e} mass_pred={mass_pred0:.3e} mass_target={mass_tgt0:.3e}"
        )
        print(
            f"[{model_label} CHECK@t0 seed=({x0_seed:.2e},{y0_seed:.2e})] "
            f"pred[min,max]=({Z0.min():.3e},{Z0.max():.3e}) "
            f"targ[min,max]=({N0.min():.3e},{N0.max():.3e})"
        )
        print(
            f"[{model_label} CHECK@t0 seed=({x0_seed:.2e},{y0_seed:.2e})] "
            f"COM_pred=({xc_p:.3e},{yc_p:.3e}) COM_target=({xc_t:.3e},{yc_t:.3e})"
        )

        # Maps at t=0 (prediction and target)
        fig_pred0, ax_pred0 = plot_map_01(
            Z0,
            X0,
            Y0,
            title=f"[{model_label}] Pred t=0  seed=({x0_seed:.2e},{y0_seed:.2e})",
            normalize_max=NORMALIZE_VIS,
            mask=mask_orig,
        )
        draw_original_box(ax_pred0, bounds_original)
        save_fig(fig_pred0)

        fig_target0, ax_target0 = plot_map_01(
            np.clip(N0, 0, 1),
            X0,
            Y0,
            title=f"[{model_label}] n0 (target) t=0  seed=({x0_seed:.2e},{y0_seed:.2e})",
            normalize_max=NORMALIZE_VIS,
            mask=mask_orig,
        )
        draw_original_box(ax_target0, bounds_original)
        save_fig(fig_target0)

        # ---- Temporal evolution: snapshots + “electron moving” maps ----
        times = times_for_seed(y0_seed, cfg_orig["sim_time"], vy, y_anode)
        frac_series = []
        Z_snapshots = []
        time_snapshots = []

        for tt in times:
            T5_t, Xt, Yt = make_grid_at_time_for_seed(
                cfg_exp,
                t_phys=float(tt),
                x0_seed=x0_seed,
                y0_seed=y0_seed,
                cfg_xy0=cfg_orig,
            )
            Zt = np.asarray(
                evaluate_on_grid_5D_seed(
                    model_tr, T5_t, nxE, nyE, batch=min(BATCH, 8192)
                )
            )
            Z_snapshots.append(Zt)
            time_snapshots.append(tt)

            # “Electron moving” map (divergent + gray outside)
            fig_move, ax_move = plot_map_electron_moving(
                Zt,
                Xt,
                Yt,
                cfg_orig,
                title=(
                    f"[{model_label}] f(t,x,y;x0,y0) t={tt:.2e}s  "
                    f"seed=({x0_seed:.2e},{y0_seed:.2e})"
                ),
                normalize_max=NORMALIZE_VIS,
            )
            draw_original_box(ax_move, bounds_original)
            save_fig(fig_move)

            mask_box = (Xt >= x0o) & (Xt <= x1o) & (Yt >= y0o) & (Yt <= y1o)

            total_mass = float(Zt.sum() + 1e-16)
            frac_below = float((Zt * (Yt < y0o)).sum()) / total_mass
            frac_series.append((tt, frac_below))

            if not mask_info_printed:
                xc, yc = center_of_mass(Zt * mask_box, Xt, Yt)
                M_box = (x1o - x0o) * (y1o - y0o) * float((Zt * mask_box).mean())
                print(
                    f"[{model_label} FRAME seed=({x0_seed:.2e},{y0_seed:.2e})] "
                    f"t={tt:.2e}s COM_y_box={yc:.3e} "
                    f"Mass_box={M_box:.3e} frac_below={frac_below:.3f}"
                )
        mask_info_printed = True

        # ---- Superposed trajectory: dense time grid to avoid “holes” ----
        if len(Z_snapshots) > 0:
            # Dense time grid between 0 and the horizon used in training
            n_traj = 80  # can increase to 60–80 for even more continuity
            t_min_traj = 0.0
            t_max_traj = float(min(t_phys_max, cfg_orig["sim_time"]))
            traj_times = np.linspace(t_min_traj, t_max_traj, n_traj)

            Z_traj_list = []
            for t_dense in traj_times:
                T5_d, _, _ = make_grid_at_time_for_seed(
                    cfg_exp,
                    t_phys=float(t_dense),
                    x0_seed=x0_seed,
                    y0_seed=y0_seed,
                    cfg_xy0=cfg_orig,
                )
                Zd = np.asarray(
                    evaluate_on_grid_5D_seed(
                        model_tr, T5_d, nxE, nyE, batch=min(BATCH, 8192)
                    )
                )
                Z_traj_list.append(Zd)

            fig_traj, _ = plot_superposed_mass_trajectory(
                Z_traj_list,
                X0,
                Y0,
                cfg_orig,
                bounds_original,
                model_label,
                x0_seed,
                y0_seed,
                normalize_vis=NORMALIZE_VIS,
            )
            if fig_traj is not None:
                save_fig(fig_traj)

        # Panel of mass evolution (only “representative” times)
        fig_mass = plot_mass_evolution_panels_for_seed(
            X0,
            Y0,
            Z_snapshots,
            time_snapshots,
            mask_orig,
            bounds_original,
            y_anode,
            x0_seed,
            y0_seed,
        )
        fig_mass.suptitle(
            f"[{model_label}] Mass evolution  seed=({x0_seed:.2e},{y0_seed:.2e})"
        )
        save_fig(fig_mass)

        # Overlaid contour plot of trajectory (as in reference code)
        if len(Z_snapshots) > 0:
            fig_cont, ax_cont = plt.subplots()
            colors = plt.cm.viridis(
                np.linspace(0.0, 1.0, len(Z_snapshots))
            )
            Xref = np.asarray(X0)
            Yref = np.asarray(Y0)
            for Zk, tk, col in zip(Z_snapshots, time_snapshots, colors):
                level = 0.5 * float(np.max(Zk))
                if level < 1e-8:
                    continue
                ax_cont.contour(
                    Xref,
                    Yref,
                    Zk,
                    levels=[level],
                    colors=[col],
                    linewidths=1.0,
                )
            draw_original_box(ax_cont, bounds_original)
            ax_cont.set_xlabel("x (m)")
            ax_cont.set_ylabel("y (m)")
            ax_cont.set_title(
                f"[{model_label}] Packet trajectory (overlaid contours)  "
                f"seed=({x0_seed:.2e},{y0_seed:.2e})"
            )
            ax_cont.grid(True, alpha=0.3)
            fig_cont.tight_layout()
            save_fig(fig_cont)

        # COM and σx, σy for “circles” plots and COM/σ(t) curves
        Xnp, Ynp = np.asarray(X0), np.asarray(Y0)
        mask_box = (Xnp >= x0o) & (Xnp <= x1o) & (Ynp >= y0o) & (Ynp <= y1o)
        com_x, com_y, sigma_x, sigma_y = [], [], [], []
        for Zt in Z_snapshots:
            Zbox = np.clip(Zt, 0.0, 1.0) * mask_box
            s = Zbox.sum()
            if s < 1e-16:
                com_x.append(np.nan)
                com_y.append(np.nan)
                sigma_x.append(np.nan)
                sigma_y.append(np.nan)
                continue
            cx = float((Zbox * Xnp).sum() / s)
            cy = float((Zbox * Ynp).sum() / s)
            com_x.append(cx)
            com_y.append(cy)
            sx = float(
                np.sqrt(((Zbox * (Xnp - cx) ** 2).sum()) / s)
            )
            sy = float(
                np.sqrt(((Zbox * (Ynp - cy) ** 2).sum()) / s)
            )
            sigma_x.append(sx)
            sigma_y.append(sy)

        # “Circles” diffusion plots
        fig_diff = plot_diffusion_circles_for_seed(
            X0,
            Y0,
            time_snapshots,
            com_x,
            com_y,
            sigma_x,
            sigma_y,
            bounds_original,
            y_anode,
            x0_seed,
            y0_seed,
        )
        fig_diff.suptitle(
            f"[{model_label}] Diffusion growth  seed=({x0_seed:.2e},{y0_seed:.2e})"
        )
        save_fig(fig_diff)

        # COM path + diffusion circles (extra)
        fig_com_path, _ = plot_com_path_with_diffusion(
            com_x,
            com_y,
            sigma_x,
            sigma_y,
            time_snapshots,
            bounds_original,
            model_label,
            x0_seed,
            y0_seed,
            y_anode,
        )
        save_fig(fig_com_path)

        # COM_y(t) and σ(t) curves for THIS seed
        fig_com, _ = plot_com_trajectory(
            np.array(time_snapshots),
            np.array(com_y),
            x0_seed,
            y0_seed,
        )
        fig_com.axes[0].set_title(
            f"[{model_label}] Center of mass trajectory (y)  seed=({x0_seed:.2e},{y0_seed:.2e})"
        )
        save_fig(fig_com)

        fig_sig, _ = plot_width_evolution(
            np.array(time_snapshots),
            np.array(sigma_x),
            np.array(sigma_y),
            x0_seed,
            y0_seed,
        )
        fig_sig.axes[0].set_title(
            f"[{model_label}] Packet broadening by diffusion  seed=({x0_seed:.2e},{y0_seed:.2e})"
        )
        save_fig(fig_sig)

        crossed = any(fr >= 0.30 for _, fr in frac_series)
        print(
            f"[{model_label}] ✅ The packet crosses the anode."
            if crossed
            else f"[{model_label}] ⚠️ [WARN] The packet did not reach the anode."
        )

    # ---- Loss curves (training logs, as in reference code) ----
    if logs is not None:
        curves = [
            ("total",  "Total Loss"),
            ("ic",     "IC Loss"),
            ("pde",    "PDE Loss"),
            ("mass",   "Mass Loss"),
            ("anchor", "Anchor Loss"),
            ("bc",     "BC (anode) Loss"),
            ("ring",   "Ring (outside) Loss"),
            ("pde_rms", "PDE Residual RMS"),
        ]
        for key_name, title in curves:
            vals = logs.get(key_name, None)
            if isinstance(vals, (list, tuple)) and len(vals) > 0:
                fig_loss, _ = plot_curve(vals, f"[{model_label}] {title}")
                save_fig(fig_loss)

    # ---- Predicted mass vs expected mass (exponential) for seed 0 ----
    tau_e = float(cfg_exp.get("tau_e", 0.0))
    if tau_e > 0.0:
        x0_seed, y0_seed = seeds_eval[0]
        mtimes = np.linspace(
            0.0,
            float(min(t_phys_max, cfg_orig["sim_time"])),
            12,  # slightly denser
        )
        Qs = [
            total_mass_on_grid_seed(
                model_tr,
                cfg_exp,
                cfg_orig,
                float(t),
                x0_seed,
                y0_seed,
                batch=min(BATCH, 8192),
            )
            for t in mtimes
        ]
        Q0 = Qs[0]
        # Normalize to see directly “predicted vs expected”
        Qs_norm = [q / (Q0 + 1e-16) for q in Qs]
        Qtarget_norm = [np.exp(-t / tau_e) for t in mtimes]

        figm, axm = plt.subplots()
        axm.plot(mtimes, Qtarget_norm, label="Expected mass  e^{-t/τ}")
        axm.plot(mtimes, Qs_norm, linestyle="--", marker="o", label="Predicted mass")
        axm.set_xlabel("Time (s)")
        axm.set_ylabel("Normalized mass  Q(t)/Q(0)")
        axm.set_title(
            f"[{model_label}] Mass predicted vs expected  seed=({x0_seed:.2e},{y0_seed:.2e})"
        )
        axm.grid(True, alpha=0.3)
        axm.legend()
        figm.tight_layout()
        save_fig(figm)

        # COM trajectory and broadening (curves) for seed 0 (extra)
        times_com = np.linspace(
            0.0,
            float(min(t_phys_max, cfg_orig["sim_time"])),
            8,
        )
        com_x_list, com_y_list, sig_x_list, sig_y_list = [], [], [], []
        for t in times_com:
            T5_t, Xg_t, Yg_t = make_grid_at_time_for_seed(
                cfg_exp, float(t), x0_seed, y0_seed, cfg_orig
            )
            Zt = np.asarray(
                evaluate_on_grid_5D_seed(
                    model_tr, T5_t, nxE, nyE, batch=min(BATCH, 8192)
                )
            )
            Xnp_t, Ynp_t = np.asarray(Xg_t), np.asarray(Yg_t)
            mask_box = (Xnp_t >= x0o) & (Xnp_t <= x1o) & (Ynp_t >= y0o) & (Ynp_t <= y1o)
            Zbox = np.clip(Zt, 0, 1) * mask_box
            s = Zbox.sum()
            if s < 1e-16:
                com_x_list.append(np.nan)
                com_y_list.append(np.nan)
                sig_x_list.append(np.nan)
                sig_y_list.append(np.nan)
                continue
            cx = float((Zbox * Xnp_t).sum() / s)
            cy = float((Zbox * Ynp_t).sum() / s)
            com_x_list.append(cx)
            com_y_list.append(cy)
            sx = float(
                np.sqrt(((Zbox * (Xnp_t - cx) ** 2).sum()) / s)
            )
            sy = float(
                np.sqrt(((Zbox * (Ynp_t - cy) ** 2).sum()) / s)
            )
            sig_x_list.append(sx)
            sig_y_list.append(sy)

        figc, _ = plot_com_trajectory(
            times_com, np.array(com_y_list), x0_seed, y0_seed
        )
        figc.axes[0].set_title(
            f"[{model_label}] COM_y(t) extra  seed=({x0_seed:.2e},{y0_seed:.2e})"
        )
        save_fig(figc)

        figw, _ = plot_width_evolution(
            times_com,
            np.array(sig_x_list),
            np.array(sig_y_list),
            x0_seed,
            y0_seed,
        )
        figw.axes[0].set_title(
            f"[{model_label}] σ(t) extra  seed=({x0_seed:.2e},{y0_seed:.2e})"
        )
        save_fig(figw)

# ==========================
# 9) Pi-test (Welch t-test)
# ==========================

def welch_ttest(x, y):
    """
    Welch's t-test for two samples x, y.
    Returns (t_stat, p_value) with two-sided p-value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")

    mean1, mean2 = x.mean(), y.mean()
    var1, var2 = x.var(ddof=1), y.var(ddof=1)

    # Standard error of the difference
    se = np.sqrt(var1 / n1 + var2 / n2)
    if se <= 0:
        return float("nan"), float("nan")

    t_stat = (mean1 - mean2) / se

    # Welch degrees of freedom
    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = (var1 ** 2) / (n1 ** 2 * (n1 - 1)) + (var2 ** 2) / (n2 ** 2 * (n2 - 1))
    if df_den <= 0:
        df = float("inf")
    else:
        df = df_num / df_den

    t_abs = float(np.abs(t_stat))

    # If df is not finite, we cannot compute a reasonable p-value
    if not np.isfinite(df):
        return float(t_stat), float("nan")

    # Try using SciPy if available
    try:
        from scipy import stats as scipy_stats
        p_val = float(2.0 * scipy_stats.t.sf(t_abs, df))
    except Exception:
        # Fallback: normal approximation N(0,1)
        from math import erf, sqrt
        z = t_abs
        # Two-sided p ≈ 2 * (1 - Phi(|z|)) = 1 - erf(|z| / sqrt(2))
        p_val = float(1.0 - erf(z / sqrt(2.0)))

    return float(t_stat), float(p_val)

# ==========================
# 10) Loss experiments + pi-test
# ==========================

def run_experiment_suite(n_repeats: int = 2,
                         patience: int = EARLY_PATIENCE,
                         min_delta: float = EARLY_MIN_DELTA,
                         do_diagnostics: bool = True):
    """
    16 configurations (MASS, ANCHOR, BC, RING on/off),
    always with IC+PDE active.
    Each configuration is trained n_repeats times (different seeds)
    to be able to run t-tests (pi-test) on pde_RMS.

    If do_diagnostics=True, full diagnostics are also run
    for the best FULL model (MASS+ANCHOR+BC+RING).
    If do_diagnostics=False, only training + pi-test are run.
    """
    extras = ["MASS", "ANCHOR", "BC", "RING"]
    metrics = {}
    combos = list(product([0, 1], repeat=len(extras)))  # 2^4 = 16

    full_label = FULL_LABEL
    best_full = {
        "pde_rms": float("inf"),
        "model": None,
        "cfg_orig": None,
        "cfg_exp": None,
        "t_phys_max": None,
        "logs": None,
    }

    total_models = len(combos)
    model_counter = 0

    for flags in combos:
        combo = dict(zip(extras, flags))
        label = "+".join([e for e, f in combo.items() if f]) or "NONE"
        metrics[label] = {"pde_rms": []}

        model_counter += 1
        print(f"\n===== MODEL {model_counter}/{total_models} :: Config={label}  extras={combo} =====")

        for rep in range(n_repeats):
            print(f"  [RUN] repeat={rep}")
            key = jax.random.PRNGKey(rep + 1234 * model_counter)

            cfg_orig = load_geometry_from_dict(GEOMETRY)
            pad_px = int(GEOMETRY["expansion"]["pixels_per_side"])
            cfg_exp = prepare_phi_and_grad(expand_domain(cfg_orig, pad_px=pad_px))

            y_anode = cfg_orig["y_bounds"][0]
            vy = float(cfg_orig["vy"])
            y_center_guess = 0.5 * (cfg_orig["y_bounds"][0] + cfg_orig["y_bounds"][1])
            t_cross_center = (
                cfg_orig["sim_time"]
                if abs(vy) < 1e-30
                else max((y_center_guess - y_anode) / abs(vy), 1e-12)
            )

            model = PINN_TXYX0Y0(width=WIDTH, depth=DEPTH, key=key, bias0=6.5)

            lam_mass   = LAM_MASS_BASE   if combo["MASS"]   else 0.0
            lam_anchor = LAM_ANCHOR_BASE if combo["ANCHOR"] else 0.0
            lam_bc     = LAM_BC_BASE     if combo["BC"]     else 0.0
            lam_ring   = LAM_RING_BASE   if combo["RING"]   else 0.0

            model_tr, logs, t_phys_max = train_curriculum_5D(
                model, cfg_orig, cfg_exp,
                key=key, epochs=EPOCHS,
                N_ic=N_IC, N_pde=N_PDE,
                N_bc_space=N_BC_SPACE, N_bc_times=N_BC_TIMES,
                N_ring_space=N_RING_SPACE, N_ring_times=N_RING_TIMES,
                N_anchor=N_ANCHOR, anchor_times=None,
                N_mass_space=N_MASS_SPACE, N_mass_times=N_MASS_TIMES, lam_mass=lam_mass,
                lr=LR, batch=BATCH,
                lam_ic=LAM_IC_BASE, lam_pde=LAM_PDE_BASE,
                lam_anchor=lam_anchor, lam_bc=lam_bc, lam_ring=lam_ring,
                ramp_epochs=RAMP, ema_beta=0.95, t_cross=t_cross_center, log_every=LOG_EVERY,
                sigma_ic=N0_SIGMA,
                early_stopping=EARLY_STOPPING, patience=patience, min_delta=min_delta,
            )

            metrics[label]["pde_rms"].append(logs["best_pde_rms"])
            print(f"    -> best_pde_RMS(rep={rep}) = {logs['best_pde_rms']:.3e} (epoch={logs['best_epoch']})")

            if label == full_label and logs["best_pde_rms"] < best_full["pde_rms"]:
                best_full["pde_rms"] = logs["best_pde_rms"]
                best_full["model"] = model_tr
                best_full["cfg_orig"] = cfg_orig
                best_full["cfg_exp"] = cfg_exp
                best_full["t_phys_max"] = t_phys_max
                best_full["logs"] = logs

    # Summary table (printed, then saved to file)
    print("\n[EXPERIMENT SUMMARY] best pde_RMS (mean ± std):")
    print(f"{'Config':30s} {'mean_pde_RMS':>14s} {'std':>10s}")
    for label, m in sorted(metrics.items()):
        arr = np.array(m["pde_rms"], dtype=float)
        mean = arr.mean()
        std  = arr.std(ddof=1) if arr.size > 1 else 0.0
        print(f"{label:30s} {mean:14.3e} {std:10.3e}")

    # Pi-test: full vs full-without-each-term
    pi_tests = []
    if full_label not in metrics:
        print("[PI-TEST] Full configuration not found, pi-test skipped.")
    else:
        full_vals = np.array(metrics[full_label]["pde_rms"], dtype=float)
        print("\n[PI-TEST] Welch t-test (pde_RMS) full vs full-without-each-term")
        for extra in extras:
            minus_terms = [e for e in extras if e != extra]
            minus_label = "+".join(minus_terms)
            if minus_label not in metrics:
                continue
            vals = np.array(metrics[minus_label]["pde_rms"], dtype=float)
            t_stat, p_val = welch_ttest(full_vals, vals)
            if np.isnan(t_stat):
                print(f"  {extra}: not enough repetitions for t-test.")
            else:
                print(
                    f"  {extra}: full ({full_label}) vs without {extra} ({minus_label}) "
                    f"-> t={t_stat:.3f}, p={p_val:.3f}"
                )
                pi_tests.append({
                    "extra": extra,
                    "full_label": full_label,
                    "minus_label": minus_label,
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "full_vals": full_vals.tolist(),
                    "minus_vals": vals.tolist(),
                })

    # Full diagnostics for the best FULL model (only if requested)
    if do_diagnostics:
        if best_full["model"] is not None:
            print(
                "\n[DIAGNOSTICS] Using the best FULL model (MASS+ANCHOR+BC+RING) "
                f"with pde_RMS={best_full['pde_rms']:.3e} to generate figures and checks."
            )
            run_full_model_diagnostics(
                best_full["model"],
                best_full["cfg_orig"],
                best_full["cfg_exp"],
                best_full["t_phys_max"],
            )
        else:
            print("[DIAGNOSTICS] No FULL model was successfully trained.")
    else:
        print("\n[DIAGNOSTICS] Skipped (Phase A: ablation + pi-test only).")

    return metrics, best_full, pi_tests

# ==========================
# 11) Saving numerical results
# ==========================

def save_experiment_results(metrics, best_full, pi_tests,
                            figs_root: str = FIGS_ROOT,
                            run_tag: str = RUN_TAG):
    """
    Saves:
      - Summary table of pde_RMS (mean ± std) in TXT.
      - Pi-test results in TXT.
      - Detailed metrics + pi-tests in JSON.
    All in the same folder as the figures: figs_root/run_tag/
    """
    import json

    outdir = os.path.join(figs_root, run_tag)
    os.makedirs(outdir, exist_ok=True)

    # Metrics summary
    summary_txt = os.path.join(outdir, "experiment_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("[EXPERIMENT SUMMARY] best pde_RMS (mean ± std):\n")
        f.write(f"{'Config':30s} {'mean_pde_RMS':>14s} {'std':>10s}\n")
        for label, m in sorted(metrics.items()):
            arr = np.array(m["pde_rms"], dtype=float)
            mean = arr.mean()
            std  = arr.std(ddof=1) if arr.size > 1 else 0.0
            f.write(f"{label:30s} {mean:14.3e} {std:10.3e}\n")

        f.write("\n[PI-TEST] Welch t-test (pde_RMS) full vs full-without-each-term\n")
        for row in pi_tests:
            f.write(
                f"  {row['extra']}: full ({row['full_label']}) vs without {row['extra']} "
                f"({row['minus_label']}) -> t={row['t_stat']:.3f}, p={row['p_value']:.3f}\n"
            )

    # Detailed JSON
    metrics_json = {
        label: {"pde_rms": [float(v) for v in m["pde_rms"]]}
        for label, m in metrics.items()
    }
    best_full_info = {
        "full_label": FULL_LABEL,
        "best_pde_rms": float(best_full["pde_rms"]),
        "best_epoch": int(best_full["logs"]["best_epoch"]) if best_full["logs"] is not None else None,
    }
    json_path = os.path.join(outdir, "metrics_pde_rms_and_pitest.json")
    with open(json_path, "w") as fj:
        json.dump(
            {
                "metrics": metrics_json,
                "best_full": best_full_info,
                "pi_tests": pi_tests,
            },
            fj,
            indent=2
        )

    print(f"[RESULTS] Numerical summary saved in:\n  {summary_txt}\n  {json_path}")

def label_to_combo(label: str):
    """
    Converts a configuration name like 'MASS+BC+RING' into a dict
    {'MASS': True/False, 'ANCHOR': True/False, 'BC': True/False, 'RING': True/False}.
    """
    extras = ["MASS", "ANCHOR", "BC", "RING"]
    if label == "NONE" or not label:
        return {e: False for e in extras}
    parts = label.split("+")
    return {e: (e in parts) for e in extras}


def load_best_ablation_label(figs_root: str = FIGS_ROOT) -> Optional[str]:
    """
    Reads the last ablation experiment (using LAST_RUN.txt) and returns
    the label of the best configuration according to the mean pde_RMS,
    excluding FULL (MASS+ANCHOR+BC+RING).
    If nothing is found, returns None.

    (This function remains available in case one wants to keep using
    the automatic criterion from Phase A, but Phase B currently uses
    by default the fixed configuration BEST_ABLATION_LABEL = 'MASS+BC+RING'.)
    """
    import json

    last_path = os.path.join(figs_root, "LAST_RUN.txt")
    if not os.path.isfile(last_path):
        print("[PHASE B] WARNING: no LAST_RUN.txt found; ablation model will be skipped.")
        return None

    with open(last_path, "r") as f:
        last_dir = f.readline().strip()

    json_path = os.path.join(last_dir, "metrics_pde_rms_and_pitest.json")
    if not os.path.isfile(json_path):
        print("[PHASE B] WARNING: no metrics_pde_rms_and_pitest.json in", last_dir)
        return None

    try:
        with open(json_path, "r") as fj:
            data = json.load(fj)
    except Exception as e:
        print(f"[PHASE B] WARNING: error reading {json_path}: {e}")
        return None

    metrics = data.get("metrics", {})
    if not metrics:
        print("[PHASE B] WARNING: empty metrics file; ablation model will be skipped.")
        return None

    best_label = None
    best_mean = None
    for label, m in metrics.items():
        # Skip the FULL model, which is explicitly trained in Phase B
        if label == FULL_LABEL:
            continue
        vals = m.get("pde_rms", [])
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        if (best_mean is None) or (mean < best_mean):
            best_mean = mean
            best_label = label

    if best_label is None:
        print("[PHASE B] WARNING: could not identify a best ablation model.")
    else:
        print(f"[PHASE B] Best ablation config (last experiment) = {best_label} "
              f"(mean_pde_RMS ≈ {best_mean:.3e})")

    return best_label

# ==========================
# 12) Main: Phase A (ablation), Phase B (FULL+best ablation) and Phase C (tuning)
# ==========================

def main_phase_a():
    """
    PHASE A: Ablation + pi-test.
    - Adjust EPOCHS and RAMP only for this phase.
    - Train the 16 configurations with the same training budget.
    - Perform Welch t-test (pi-test) ONLY with these results.
    - Do not run physical diagnostics (left for Phase B/C).
    """
    global EPOCHS, RAMP

    # Controlled budget for the ablation experiment
    EPOCHS = 100     # all configurations, same number of epochs
    RAMP   = 80      # PDE/MASS losses ramp almost fully within these 100 epochs

    print(f"[PHASE A] Ablation + pi-test with EPOCHS={EPOCHS}, RAMP={RAMP}")
    print(f"[PHASE A] n_repeats=2, early_stopping={EARLY_STOPPING}, patience={EARLY_PATIENCE}")

    metrics, best_full, pi_tests = run_experiment_suite(
        n_repeats=2,
        patience=EARLY_PATIENCE,
        min_delta=EARLY_MIN_DELTA,
        do_diagnostics=False,   # do NOT run physical diagnostics here
    )

    # Save numerical summary (table + pi-test)
    save_experiment_results(
        metrics, best_full, pi_tests,
        figs_root=FIGS_ROOT,
        run_tag=RUN_TAG,
    )

    # Save figures (if any; typically few or none in Phase A)
    finalize_figures(FIGS_ROOT, RUN_TAG)


def main_phase_b():
    """
    PHASE B: Comparison FULL vs best ablation model.
    - Train the FULL model (MASS+ANCHOR+BC+RING) with long training (EPOCHS=600).
    - Also train the ablation model chosen a priori: MASS+BC+RING.
      This choice is fixed and does not depend on Phase A having been run.
    - Run the same physical diagnostics for both models.
    - At the end, print which one has the best best_pde_RMS.
    """
    print(
        f"[PHASE B] Training FULL model ({FULL_LABEL}) and fixed ablation model "
        f"({BEST_ABLATION_LABEL}) with EPOCHS={EPOCHS}, RAMP={RAMP}"
    )

    # Geometry and domain (shared by both models)
    cfg_orig = load_geometry_from_dict(GEOMETRY)
    pad_px = int(GEOMETRY["expansion"]["pixels_per_side"])
    cfg_exp = prepare_phi_and_grad(expand_domain(cfg_orig, pad_px=pad_px))

    # Estimated time horizon to cross the anode
    y_anode = cfg_orig["y_bounds"][0]
    vy = float(cfg_orig["vy"])
    y_center_guess = 0.5 * (cfg_orig["y_bounds"][0] + cfg_orig["y_bounds"][1])
    t_cross_center = (
        cfg_orig["sim_time"]
        if abs(vy) < 1e-30
        else max((y_center_guess - y_anode) / abs(vy), 1e-12)
    )

    # ==========================
    # 1) FULL model
    # ==========================
    print("[PHASE B] Training FULL model (MASS+ANCHOR+BC+RING)...")

    key_full = jax.random.PRNGKey(42)
    model_full = PINN_TXYX0Y0(width=WIDTH, depth=DEPTH, key=key_full, bias0=6.5)

    lam_mass_full   = LAM_MASS_BASE
    lam_anchor_full = LAM_ANCHOR_BASE
    lam_bc_full     = LAM_BC_BASE
    lam_ring_full   = LAM_RING_BASE

    model_tr_full, logs_full, t_phys_max_full = train_curriculum_5D(
        model_full, cfg_orig, cfg_exp,
        key=key_full, epochs=EPOCHS,
        N_ic=N_IC, N_pde=N_PDE,
        N_bc_space=N_BC_SPACE, N_bc_times=N_BC_TIMES,
        N_ring_space=N_RING_SPACE, N_ring_times=N_RING_TIMES,
        N_anchor=N_ANCHOR, anchor_times=None,
        N_mass_space=N_MASS_SPACE, N_mass_times=N_MASS_TIMES, lam_mass=lam_mass_full,
        lr=LR, batch=BATCH,
        lam_ic=LAM_IC_BASE, lam_pde=LAM_PDE_BASE,
        lam_anchor=lam_anchor_full, lam_bc=lam_bc_full, lam_ring=lam_ring_full,
        ramp_epochs=RAMP, ema_beta=0.95, t_cross=t_cross_center, log_every=LOG_EVERY,
        sigma_ic=N0_SIGMA,
        early_stopping=False,              # <- 600 epochs no matter what
        patience=EARLY_PATIENCE,
        min_delta=EARLY_MIN_DELTA,
    )

    print(
        f"[PHASE B] FULL: best_pde_RMS={logs_full['best_pde_rms']:.3e} "
        f"at epoch={logs_full['best_epoch']}"
    )

    # Full diagnostics for FULL
    run_full_model_diagnostics(
        model_tr_full, cfg_orig, cfg_exp, t_phys_max_full, logs_full,
        model_label="FULL"
    )

    # ==========================
    # 2) Fixed ablation model MASS+BC+RING
    # ==========================
    model_tr_ab = None
    logs_ab = None

    best_ablation_label = BEST_ABLATION_LABEL
    combo = label_to_combo(best_ablation_label)
    lam_mass_ab   = LAM_MASS_BASE   if combo["MASS"]   else 0.0
    lam_anchor_ab = LAM_ANCHOR_BASE if combo["ANCHOR"] else 0.0
    lam_bc_ab     = LAM_BC_BASE     if combo["BC"]     else 0.0
    lam_ring_ab   = LAM_RING_BASE   if combo["RING"]   else 0.0

    print(f"[PHASE B] Training ablation model '{best_ablation_label}' "
          f"with EPOCHS={EPOCHS}, RAMP={RAMP}...")

    key_ab = jax.random.PRNGKey(123)
    model_ab = PINN_TXYX0Y0(width=WIDTH, depth=DEPTH, key=key_ab, bias0=6.5)

    model_tr_ab, logs_ab, t_phys_max_ab = train_curriculum_5D(
        model_ab, cfg_orig, cfg_exp,
        key=key_ab, epochs=EPOCHS,
        N_ic=N_IC, N_pde=N_PDE,
        N_bc_space=N_BC_SPACE, N_bc_times=N_BC_TIMES,
        N_ring_space=N_RING_SPACE, N_ring_times=N_RING_TIMES,
        N_anchor=N_ANCHOR, anchor_times=None,
        N_mass_space=N_MASS_SPACE, N_mass_times=N_MASS_TIMES, lam_mass=lam_mass_ab,
        lr=LR, batch=BATCH,
        lam_ic=LAM_IC_BASE, lam_pde=LAM_PDE_BASE,
        lam_anchor=lam_anchor_ab, lam_bc=lam_bc_ab, lam_ring=lam_ring_ab,
        ramp_epochs=RAMP, ema_beta=0.95, t_cross=t_cross_center, log_every=LOG_EVERY,
        sigma_ic=N0_SIGMA,
        early_stopping=False,              # <- also 600 epochs no matter what
        patience=EARLY_PATIENCE,
        min_delta=EARLY_MIN_DELTA,
    )

    print(
        f"[PHASE B] '{best_ablation_label}': best_pde_RMS={logs_ab['best_pde_rms']:.3e} "
        f"at epoch={logs_ab['best_epoch']}"
    )

    # Same diagnostics suite for the best ablation model
    run_full_model_diagnostics(
        model_tr_ab, cfg_orig, cfg_exp, t_phys_max_ab, logs_ab,
        model_label=best_ablation_label
    )

    # ==========================
    # 3) Final comparison FULL vs ablation model
    # ==========================
    full_val = float(logs_full["best_pde_rms"])
    abl_val  = float(logs_ab["best_pde_rms"])
    if abl_val < full_val:
        print(f"[PHASE B] RESULT: ablation model '{best_ablation_label}' "
              f"is better (pde_RMS={abl_val:.3e}) than FULL (pde_RMS={full_val:.3e}).")
    else:
        print(f"[PHASE B] RESULT: FULL model is still better "
              f"(pde_RMS={full_val:.3e}) than '{best_ablation_label}' (pde_RMS={abl_val:.3e}).")

    # Save all figures (FULL + ablation) in the same folder
    finalize_figures(FIGS_ROOT, RUN_TAG)


def main_phase_c():
    """
    PHASE C: Hyperparameter tuning for the best ablation model (MASS+BC+RING).
    - Keep the MASS+BC+RING loss combination fixed.
    - Explore several LR and width combinations.
    - Train each configuration for the same number of epochs.
    - Run physical diagnostics only for the best configuration.
    """
    print(f"[PHASE C] Hyperparameter tuning for ablation model '{BEST_ABLATION_LABEL}'")

    # Geometry and domain
    cfg_orig = load_geometry_from_dict(GEOMETRY)
    pad_px = int(GEOMETRY["expansion"]["pixels_per_side"])
    cfg_exp = prepare_phi_and_grad(expand_domain(cfg_orig, pad_px=pad_px))

    # Estimated time horizon
    y_anode = cfg_orig["y_bounds"][0]
    vy = float(cfg_orig["vy"])
    y_center_guess = 0.5 * (cfg_orig["y_bounds"][0] + cfg_orig["y_bounds"][1])
    t_cross_center = (
        cfg_orig["sim_time"]
        if abs(vy) < 1e-30
        else max((y_center_guess - y_anode) / abs(vy), 1e-12)
    )

    # Hyperparameter grid
    lr_grid = [1e-4, 3e-4, 5e-4]
    width_grid = [256, 320]
    epochs_tune = 400
    ramp_tune = 400

    combo = label_to_combo(BEST_ABLATION_LABEL)
    lam_mass   = LAM_MASS_BASE   if combo["MASS"]   else 0.0
    lam_anchor = LAM_ANCHOR_BASE if combo["ANCHOR"] else 0.0
    lam_bc     = LAM_BC_BASE     if combo["BC"]     else 0.0
    lam_ring   = LAM_RING_BASE   if combo["RING"]   else 0.0

    best_pde = float("inf")
    best_conf = None
    best_model = None
    best_logs = None
    best_t_phys_max = None

    total_configs = len(lr_grid) * len(width_grid)
    idx_conf = 0

    for lr_hp in lr_grid:
        for width_hp in width_grid:
            idx_conf += 1
            print(f"\n[PHASE C] Config {idx_conf}/{total_configs}: lr={lr_hp:.1e}, width={width_hp}")

            key_hp = jax.random.PRNGKey(1000 + idx_conf)
            model_hp = PINN_TXYX0Y0(width=width_hp, depth=DEPTH, key=key_hp, bias0=6.5)

            model_tr_hp, logs_hp, t_phys_max_hp = train_curriculum_5D(
                model_hp, cfg_orig, cfg_exp,
                key=key_hp, epochs=epochs_tune,
                N_ic=N_IC, N_pde=N_PDE,
                N_bc_space=N_BC_SPACE, N_bc_times=N_BC_TIMES,
                N_ring_space=N_RING_SPACE, N_ring_times=N_RING_TIMES,
                N_anchor=N_ANCHOR, anchor_times=None,
                N_mass_space=N_MASS_SPACE, N_mass_times=N_MASS_TIMES, lam_mass=lam_mass,
                lr=lr_hp, batch=BATCH,
                lam_ic=LAM_IC_BASE, lam_pde=LAM_PDE_BASE,
                lam_anchor=lam_anchor, lam_bc=lam_bc, lam_ring=lam_ring,
                ramp_epochs=ramp_tune, ema_beta=0.95, t_cross=t_cross_center, log_every=LOG_EVERY,
                sigma_ic=N0_SIGMA,
                early_stopping=False,
                patience=EARLY_PATIENCE,
                min_delta=EARLY_MIN_DELTA,
            )

            pde_val = float(logs_hp["best_pde_rms"])
            print(f"[PHASE C]  -> best_pde_RMS={pde_val:.3e} (epoch={logs_hp['best_epoch']})")

            if pde_val < best_pde:
                best_pde = pde_val
                best_conf = {"lr": lr_hp, "width": width_hp}
                best_model = model_tr_hp
                best_logs = logs_hp
                best_t_phys_max = t_phys_max_hp

    if best_conf is not None:
        print("\n[PHASE C] Best configuration found:")
        print(f"  lr={best_conf['lr']:.1e}, width={best_conf['width']} -> pde_RMS={best_pde:.3e}")

        # Diagnostics for the best tuned model
        run_full_model_diagnostics(
            best_model,
            cfg_orig,
            cfg_exp,
            best_t_phys_max,
            best_logs,
            model_label=f"{BEST_ABLATION_LABEL}_TUNED",
        )
    else:
        print("[PHASE C] No valid hyperparameter configuration found.")

    finalize_figures(FIGS_ROOT, RUN_TAG)

def main_phase_d():
    """
    PHASE D: Long MASS+BC+RING (1200 epochs, RAMP=800).
    Same as the ablation part of Phase B but only for MASS+BC+RING
    and with longer training.
    """
    # Phase D specific hyperparameters
    epochs_d = 1200
    ramp_d   = 800

    best_ablation_label = BEST_ABLATION_LABEL  # "MASS+BC+RING"
    print(f"[PHASE D] Training ablation model '{best_ablation_label}' "
          f"with EPOCHS={epochs_d}, RAMP={ramp_d}")

    # Geometry and domain (same as in Phase B)
    cfg_orig = load_geometry_from_dict(GEOMETRY)
    pad_px   = int(GEOMETRY["expansion"]["pixels_per_side"])
    cfg_exp  = prepare_phi_and_grad(expand_domain(cfg_orig, pad_px=pad_px))

    # Estimated time horizon to cross the anode
    y_anode = cfg_orig["y_bounds"][0]
    vy      = float(cfg_orig["vy"])
    y_center_guess = 0.5 * (cfg_orig["y_bounds"][0] + cfg_orig["y_bounds"][1])
    t_cross_center = (
        cfg_orig["sim_time"]
        if abs(vy) < 1e-30
        else max((y_center_guess - y_anode) / abs(vy), 1e-12)
    )

    # MASS+BC+RING combination (same as in B)
    combo       = label_to_combo(best_ablation_label)
    lam_mass    = LAM_MASS_BASE   if combo["MASS"]   else 0.0
    lam_anchor  = LAM_ANCHOR_BASE if combo["ANCHOR"] else 0.0
    lam_bc      = LAM_BC_BASE     if combo["BC"]     else 0.0
    lam_ring    = LAM_RING_BASE   if combo["RING"]   else 0.0

    # Model and seed (you can keep the same as in Phase B)
    key_d   = jax.random.PRNGKey(123)
    model_d = PINN_TXYX0Y0(width=WIDTH, depth=DEPTH, key=key_d, bias0=6.5)

    # Long MASS+BC+RING training
    model_tr_d, logs_d, t_phys_max_d = train_curriculum_5D(
        model_d, cfg_orig, cfg_exp,
        key=key_d, epochs=epochs_d,
        N_ic=N_IC, N_pde=N_PDE,
        N_bc_space=N_BC_SPACE, N_bc_times=N_BC_TIMES,
        N_ring_space=N_RING_SPACE, N_ring_times=N_RING_TIMES,
        N_anchor=N_ANCHOR, anchor_times=None,
        N_mass_space=N_MASS_SPACE, N_mass_times=N_MASS_TIMES, lam_mass=lam_mass,
        lr=LR, batch=BATCH,
        lam_ic=LAM_IC_BASE, lam_pde=LAM_PDE_BASE,
        lam_anchor=lam_anchor, lam_bc=lam_bc, lam_ring=lam_ring,
        ramp_epochs=ramp_d, ema_beta=0.95, t_cross=t_cross_center,
        log_every=LOG_EVERY,
        sigma_ic=N0_SIGMA,
        early_stopping=False,          # <- 1200 epochs no matter what
        patience=EARLY_PATIENCE,
        min_delta=EARLY_MIN_DELTA,
    )

    print(
        f"[PHASE D] '{best_ablation_label}': best_pde_RMS={logs_d['best_pde_rms']:.3e} "
        f"at epoch={logs_d['best_epoch']}"
    )

    # Full physical diagnostics, same as in Phase B
    run_full_model_diagnostics(
        model_tr_d,
        cfg_orig,
        cfg_exp,
        t_phys_max_d,
        logs_d,
        model_label=f"{best_ablation_label}_E1200_R800",
    )

    # Save all figures from Phase D
    finalize_figures(FIGS_ROOT, RUN_TAG)

def main():
    """
    Phase selector:
      - Phase A: ablation experiment + pi-test (16 combos, short training).
      - Phase B: FULL model + fixed best ablation model (MASS+BC+RING), long training + diagnostics.
      - Phase C: hyperparameter tuning for the best ablation model (MASS+BC+RING).
      - Phase D: long MASS+BC+RING (1200 epochs, RAMP=800) with full diagnostics.

    Selected via environment variable PINN_PHASE:
      PINN_PHASE=A  -> Phase A
      PINN_PHASE=B  -> Phase B
      PINN_PHASE=C  -> Phase C
      PINN_PHASE=D  -> Phase D

    If PINN_PHASE is not defined, Phase A is run by default.
    """
    phase = os.environ.get("PINN_PHASE", "A").upper()
    print(f"[MAIN] PINN_PHASE={phase}")

    if phase == "A":
        main_phase_a()
    elif phase == "B":
        main_phase_b()
    elif phase == "C":
        main_phase_c()
    elif phase == "D":
        main_phase_d()
    else:
        raise SystemExit(f"Unknown PINN_PHASE value: '{phase}'. Use 'A', 'B', 'C' or 'D'.")


if __name__ == "__main__":
    main()
