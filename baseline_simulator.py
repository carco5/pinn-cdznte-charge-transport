# baseline_simulator.py
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_geometry, load_compute_laplace_equation

# ============================
# Configuración de visualización
# ============================

# Si es True, cada frame se reescala dividiendo por su máximo local.
# La barra de color siempre va de 0 a 1, así todas las figuras son comparables.
NORMALIZE_VIS = True
CMAP_DIVERGENT = "coolwarm"   # colormap divergente tipo coolwarm

# Padding (en número de píxeles) para construir un dominio extendido
# y poder mostrar la "malla blanca" con gris alrededor, igual que el PINN.
PAD_PX = 40

# Tamaño (σ) del paquete gaussiano inicial en número de celdas.
# Antes era sigma=2 (electrón muy pequeño). Con 6 se ve claramente más grande.
GAUSS_SIGMA_PIX = 6.0


# ============================
# Funciones auxiliares
# ============================

def grad_centered(arr, dx, dy):
    """Gradiente centrado 2D con condiciones Neumann."""
    gx = jnp.zeros_like(arr); gy = jnp.zeros_like(arr)
    gx = gx.at[1:-1, :].set((arr[2:, :] - arr[:-2, :]) / (2*dx))
    gy = gy.at[:, 1:-1].set((arr[:, 2:] - arr[:, :-2]) / (2*dy))
    gx = gx.at[0, :].set(gx[1, :]); gx = gx.at[-1, :].set(gx[-2, :])
    gy = gy.at[:, 0].set(gy[:, 1]); gy = gy.at[:, -1].set(gy[:, -2])
    return gx, gy


def divergence_mu_nE(n, Ex, Ey, mu, dx, dy):
    """∇·(μ n E) usando esquema centrado simple."""
    n0 = n[0]
    Ex0, Ey0 = Ex[0], Ey[0]

    # flujo en x
    Fx = mu * Ex0 * n0
    divx = jnp.zeros_like(n0)
    divx = divx.at[1:-1, :].set((Fx[2:, :] - Fx[:-2, :]) / (2*dx))

    # flujo en y
    Fy = mu * Ey0 * n0
    divy = jnp.zeros_like(n0)
    divy = divy.at[:, 1:-1].set((Fy[:, 2:] - Fy[:, :-2]) / (2*dy))

    return (divx + divy)[None, ...]


# ============================
# Simulador drift-only
# ============================

def simulator_drift(phi, mu_e, dx, dy, dt, steps, init_pos=(75, 200), sigma=GAUSS_SIGMA_PIX):
    # Calcular campo eléctrico E = ∇phi
    gpx, gpy = grad_centered(phi[0], dx, dy)
    Ex, Ey = (gpx)[None, ...], (gpy)[None, ...]

    # Inicialización: gaussiana alrededor de init_pos (en coordenadas de celda)
    nx, ny = phi.shape[1], phi.shape[2]
    X, Y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")

    # sigma en píxeles (más grande para que el electrón se vea más gordo)
    n0 = jnp.exp(-(((X - init_pos[0]) ** 2 + (Y - init_pos[1]) ** 2) /
                   (2.0 * sigma ** 2)))
    n = n0[None, ...]

    frames = [n]

    for _ in range(steps):
        rhs = -divergence_mu_nE(n, Ex, Ey, mu_e, dx, dy)
        n_next = n + dt * rhs
        # no negatividad y cota superior 1
        n_next = jnp.clip(n_next, a_min=0.0, a_max=1.0)
        n = n_next
        frames.append(n)

    return jnp.stack(frames)


# ============================
# Helpers de visualización
# ============================

def draw_original_box(ax, x_bounds, y_bounds):
    """Dibuja el rectángulo blanco del detector original (como en el PINN)."""
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
    Construye una malla extendida en torno al detector original:
      - tamaño nx_ext = nx + 2*pad_px
      - tamaño ny_ext = ny + 2*pad_px
    Devuelve X_ext, Y_ext con shape (nx_ext, ny_ext).
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

    # indexing="xy" y luego transponer para tener shape (nx_ext, ny_ext)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return X.T, Y.T


def plot_electron_frame(Z, X_ext, Y_ext, config, title=""):
    """
    Dibuja un frame 2D con el mismo estilo que el PINN:
      - colormap divergente (coolwarm),
      - barra de color fija [0,1] para todas las figuras,
      - píxeles cuadrados (aspect='equal'),
      - detector original enmarcado por un rectángulo blanco,
      - región fuera del detector en gris.
    Z se simula sólo en el dominio original y se inserta en el dominio extendido.
    """
    # Z viene con shape (nx, ny)
    Z = np.asarray(jnp.clip(Z, 0.0, 1.0))
    if NORMALIZE_VIS and Z.max() > 0:
        Z = Z / float(Z.max())

    nx0, ny0 = Z.shape
    nx_ext, ny_ext = X_ext.shape

    if nx_ext < nx0 or ny_ext < ny0:
        raise ValueError("El dominio extendido es más pequeño que el original.")

    # Insertamos Z en el centro del dominio extendido
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

    # Máscara: dentro del detector (True) vs fuera (False)
    mask_inside = (Xn >= x0o) & (Xn <= x1o) & (Yn >= y0o) & (Yn <= y1o)

    # Para imshow transponemos
    Z_plot = Z_ext.T
    mask_plot = (~mask_inside).T  # True = fuera ⇒ gris

    Z_masked = np.ma.array(Z_plot, mask=mask_plot)

    cmap = plt.get_cmap(CMAP_DIVERGENT).copy()
    cmap.set_bad(color="0.5")  # gris para fuera de la caja original

    fig, ax = plt.subplots()
    im = ax.imshow(
        Z_masked,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",  # electrónica circular y escala coherente
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    # Dibujar el rectángulo blanco del detector original
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

    # Dimensiones del dominio original
    nx = phi.shape[1]
    ny = phi.shape[2]

    dx = (config["x_bounds"][1] - config["x_bounds"][0]) / nx
    dy = (config["y_bounds"][1] - config["y_bounds"][0]) / ny
    dt = config["dt"]
    mu_e = config["mu_e"]

    # Resolver φ si no existe el .npy
    phi = load_compute_laplace_equation(phi, str(HERE / "phi.npy"))

    # Construir malla extendida para visualización (igual que en el PINN: padding de 40 px)
    X_ext, Y_ext = build_extended_coords(
        config["x_bounds"],
        config["y_bounds"],
        dx,
        dy,
        nx,
        ny,
        pad_px=PAD_PX,
    )

    # Simulación drift-only
    steps = 10_000
    frames = simulator_drift(
        phi,
        mu_e,
        dx,
        dy,
        dt,
        steps=steps,
        init_pos=(75, 50),          # misma posición que tenías
        sigma=GAUSS_SIGMA_PIX,      # <-- electrón más grande
    )

    # Visualización en varios instantes (misma escala + colormap divergente + gris alrededor)
    for t in [0, 100, 200, 300, 400, 2_000, 10_000]:
        fig, ax = plot_electron_frame(
            frames[t, 0],
            X_ext,
            Y_ext,
            config,
            title=f"Evento puntual n (step={t}, t={t*dt:.2e} s)",
        )

    plt.show()
