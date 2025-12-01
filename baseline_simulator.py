# baseline_simulator.py
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_geometry, load_compute_laplace_equation

# ============================
# Visualization settings
# ============================

# If True, each frame is rescaled by dividing by its local maximum.
# The color bar always goes from 0 to 1, so all figures are comparable.
NORMALIZE_VIS = True
CMAP_DIVERGENT = "coolwarm"   # divergent colormap like coolwarm

# Padding (in number of pixels) to build an extended domain
# and be able to show the "white mesh" with gray around it, same as the PINN.
PAD_PX = 40

# Size (σ) of the initial Gaussian packet in number of cells.
# Previously sigma=2 (very small electron). With 6 it looks clearly bigger.
GAUSS_SIGMA_PIX = 6.0


# ============================
# Auxiliary functions
# ============================

def grad_centered(arr, dx, dy):
    """2D centered gradient with Neumann boundary conditions."""
    gx = jnp.zeros_like(arr); gy = jnp.zeros_like(arr)
    gx = gx.at[1:-1, :].set((arr[2:, :] - arr[:-2, :]) / (2*dx))
    gy = gy.at[:, 1:-1].set((arr[:, 2:] - arr[:, :-2]) / (2*dy))
    gx = gx.at[0, :].set(gx[1, :]); gx = gx.at[-1, :].set(gx[-2, :])
    gy = gy.at[:, 0].set(gy[:, 1]); gy = gy.at[:, -1].set(gy[:, -2])
    return gx, gy


def divergence_mu_nE(n, Ex, Ey, mu, dx, dy):
    """∇·(μ n E) using a simple centered scheme."""
    n0 = n[0]
    Ex0, Ey0 = Ex[0], Ey[0]

    # flux in x
    Fx = mu * Ex0 * n0
    divx = jnp.zeros_like(n0)
    divx = divx.at[1:-1, :].set((Fx[2:, :] - Fx[:-2, :]) / (2*dx))

    # flux in y
    Fy = mu * Ey0 * n0
    divy = jnp.zeros_like(n0)
    divy = divy.at[:, 1:-1].set((Fy[:, 2:] - Fy[:, :-2]) / (2*dy))

    return (divx + divy)[None, ...]


# ============================
# Drift-only simulator
# ============================

def simulator_drift(phi, mu_e, dx, dy, dt, steps, init_pos=(75, 200), sigma=GAUSS_SIGMA_PIX):
    # Compute electric field E = ∇phi
    gpx, gpy = grad_centered(phi[0], dx, dy)
    Ex, Ey = (gpx)[None, ...], (gpy)[None, ...]

    # Initialization: Gaussian around init_pos (in cell coordinates)
    nx, ny = phi.shape[1], phi.shape[2]
    X, Y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")

    # sigma in pixels (larger so the electron looks fatter)
    n0 = jnp.exp(-(((X - init_pos[0]) ** 2 + (Y - init_pos[1]) ** 2) /
                   (2.0 * sigma ** 2)))
    n = n0[None, ...]

    frames = [n]

    for _ in range(steps):
        rhs = -divergence_mu_nE(n, Ex, Ey, mu_e, dx, dy)
        n_next = n + dt * rhs
        # non-negativity and upper bound 1
        n_next = jnp.clip(n_next, a_min=0.0, a_max=1.0)
        n = n_next
        frames.append(n)

    return jnp.stack(frames)


# ============================
# Visualization helpers
# ============================

def draw_original_box(ax, x_bounds, y_bounds):
    """Draws the white rectangle of the original detector (as in the PINN)."""
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    ax.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        color="white",
        linewidth=1.5,
    )


def build_extended_coords(x_bounds, y_bounds, dx, dy, nx, ny, pad_px=PAD_PX):
    """
    Builds an extended mesh around the original detector:
      - size nx_ext = nx + 2*pad_px
      - size ny_ext = ny + 2*pad_px
    Returns X_ext, Y_ext with shape (nx_ext, ny_ext).
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    nx_ext = nx + 2 * pad_px
    ny_ext = ny + 2 * pad_px

    x0_ext = x_min - pad_px * dx
    y0_ext = y_min - pad_px * dy
    x1_ext = x_max + pad_px * dx
    y1_ext = y_max + pad_px * dy

    xs = np.linspace(x0_ext, x1_ext, nx_ext, endpoint=False)
    ys = np.linspace(y0_ext, y1_ext, ny_ext, endpoint=False)

    # indexing="xy" and then transpose to have shape (nx_ext, ny_ext)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return X.T, Y.T


def plot_electron_frame(Z, X_ext, Y_ext, config, title=""):
    """
    Draws a 2D frame with the same style as the PINN:
      - divergent colormap (coolwarm),
      - fixed color bar [0,1] for all figures,
      - square pixels (aspect='equal'),
      - original detector framed by a white rectangle,
      - region outside the detector in gray.
    Z is simulated only in the original domain and is inserted into the extended domain.
    """
    # Z comes with shape (nx, ny)
    Z = np.asarray(jnp.clip(Z, 0.0, 1.0))
    if NORMALIZE_VIS and Z.max() > 0:
        Z = Z / float(Z.max())

    nx0, ny0 = Z.shape
    nx_ext, ny_ext = X_ext.shape

    if nx_ext < nx0 or ny_ext < ny0:
        raise ValueError("The extended domain is smaller than the original.")

    # Insert Z in the center of the extended domain
    Z_ext = np.zeros((nx_ext, ny_ext), dtype=float)
    off_x = (nx_ext - nx0) // 2
    off_y = (ny_ext - ny0) // 2
    Z_ext[off_x:off_x + nx0, off_y:off_y + ny0] = Z

    Xn = np.asarray(X_ext)
    Yn = np.asarray(Y_ext)
    extent = [Xn.min(), Xn.max(), Yn.min(), Yn.max()]

    x_bounds = config["x_bounds"]
    y_bounds = config["y_bounds"]
    x0o, x1o = x_bounds
    y0o, y1o = y_bounds

    # Mask: inside detector (True) vs outside (False)
    mask_inside = (Xn >= x0o) & (Xn <= x1o) & (Yn >= y0o) & (Yn <= y1o)

    # For imshow we transpose
    Z_plot = Z_ext.T
    mask_plot = (~mask_inside).T  # True = outside ⇒ gray

    Z_masked = np.ma.array(Z_plot, mask=mask_plot)

    cmap = plt.get_cmap(CMAP_DIVERGENT).copy()
    cmap.set_bad(color="0.5")  # gray for outside the original box

    fig, ax = plt.subplots()
    im = ax.imshow(
        Z_masked,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",  # circular electron cloud and consistent scaling
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    # Draw the white rectangle of the original detector
    draw_original_box(ax, x_bounds, y_bounds)

    fig.tight_layout()
    return fig, ax


# ============================
# Main
# ============================

if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    GEOM_PATH = HERE / "geometry.yaml"

    print("LOADED GEOMETRY")
    phi, anode, drift, cathode, config = load_geometry(str(GEOM_PATH))

    # Dimensions of the original domain
    nx = phi.shape[1]
    ny = phi.shape[2]

    dx = (config["x_bounds"][1] - config["x_bounds"][0]) / nx
    dy = (config["y_bounds"][1] - config["y_bounds"][0]) / ny
    dt = config["dt"]
    mu_e = config["mu_e"]

    # Solve φ if the .npy does not exist
    phi = load_compute_laplace_equation(phi, str(HERE / "phi.npy"))

    # Build extended mesh for visualization (same as in the PINN: padding of 40 px)
    X_ext, Y_ext = build_extended_coords(
        config["x_bounds"],
        config["y_bounds"],
        dx,
        dy,
        nx,
        ny,
        pad_px=PAD_PX,
    )

    # Drift-only simulation
    steps = 10_000
    frames = simulator_drift(
        phi,
        mu_e,
        dx,
        dy,
        dt,
        steps=steps,
        init_pos=(75, 50),          # same position as you had
        sigma=GAUSS_SIGMA_PIX,      # <-- bigger electron
    )

    # Visualization at several time instants (same scale + divergent colormap + gray around)
    for t in [0, 100, 200, 300, 400, 2_000, 10_000]:
        fig, ax = plot_electron_frame(
            frames[t, 0],
            X_ext,
            Y_ext,
            config,
            title=f"Point-like event n (step={t}, t={t*dt:.2e} s)",
        )

    plt.show()
