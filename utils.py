import jax
import jax.numpy as jnp
import yaml
import time
import os


@jax.jit
def jacobi_method_2d(
    geometry: jnp.ndarray, iterations: int = 100_000
) -> tuple[jnp.ndarray, float]:
    """Solve the 2D Laplace using Jacobi method for a fixed number of iterations and returns the solved Laplace equation and the max error in the last iteration

    Args:
        geometry (jnp.ndarray): Initial geometry where the fixed values are not NAN and everything else is NAN
        iterations (int, optional): Number of iterations to run the loop. Defaults to 1e6.

    Returns:
        tuple[jnp.ndarray, float]: Both the new calculated geometry and the error in tha las round
    """
    dirichlet_mask = ~jnp.isnan(geometry)
    fixed_values = geometry.copy()
    geometry = jnp.where(dirichlet_mask, geometry, 0.0)

    def update_step(i, state):
        u, _ = state

        # Jacobi update on interior points
        u_new = u.at[1:-1, 1:-1].set(
            0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])
        )

        # Neumann boundary conditions (copy adjacent values)
        u_new = u_new.at[0, :].set(u_new[1, :])  # Top
        u_new = u_new.at[-1, :].set(u_new[-2, :])  # Bottom
        u_new = u_new.at[:, 0].set(u_new[:, 1])  # Left
        u_new = u_new.at[:, -1].set(u_new[:, -2])  # Right

        # Apply Dirichlet conditions
        u_new = jnp.where(dirichlet_mask, fixed_values, u_new)

        error = jnp.max(jnp.abs(u_new - u))
        return u_new, error

    init_state = (geometry, 0.0)
    final_u, max_error = jax.lax.fori_loop(0, iterations, update_step, init_state)

    return final_u, max_error


def load_compute_laplace_equation(geometry, save_path, iterations=100_000):
    start = time.time()
    geometry_path = save_path

    if not os.path.isfile(geometry_path):
        geometry, error = jax.vmap(jacobi_method_2d, in_axes=(0, None))(
            geometry, iterations
        )
        print(
            f"Computed Laplace in {round(time.time() - start)} seconds with {error.mean():.2e} average error"
        )
        jnp.save(geometry_path, geometry)
    else:
        new_geometry = jnp.load(geometry_path, "r")
        if new_geometry.shape != geometry.shape:
            print("WRONG SHAPES FROM SAVED FILE")
            geometry, error = jax.vmap(jacobi_method_2d, in_axes=(0, None))(
                geometry, iterations
            )
            print(error)
            print(
                f"Computed Laplace in {round(time.time() - start)} seconds with {error.mean():.2e} average error"
            )
            jnp.save(geometry_path, geometry)
        else:
            print("LOADED GEOMETRY")
            geometry = new_geometry

    return geometry


def _load_yaml(path: str) -> dict[any, any]:
    """Load the YAML file

    Args:
        path (str): path to the YAML

    Returns:
        dict[any, any]: the containing data
    """
    # --- ORIGINAL (podía fallar en Windows por cp1252):
    # with open(path, "r") as f:
    #     return yaml.safe_load(f)

    # --- CORREGIDO: forzar UTF-8 para evitar UnicodeDecodeError en Windows ---
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except UnicodeDecodeError:
        # Fallback por si el archivo estuviera en otra codificación rara.
        # 'errors="replace"' evita crashear (reemplaza caracteres ilegales).
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return yaml.safe_load(f)



def load_geometry(
    path: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[any, any]]:
    """Load the geometry for the electric field and the potential fields.

    NOTE: For now, the drifts are slightly different to the others as they are 3 together in a single electrode

    NOTE: The return structure is NAN everywhere except on the electrodes, where depends on the value.

    Args:
        path (str): path to the config file with all the configuration

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: the initialization of the electric field, the anode potentials, the drift potentials and the cathode potentials, and all the data information in a dict.
    """
    data = _load_yaml(path)

    dimensions = data["dimensions"]
    electrode_info = data["electrodes"]
    phyiscs_data = data["physics"]

    # TODO: Control whether is 2D or 3D, or make a separate function/module that does the same in 3D

    nx, ny = dimensions["steps"]
    x_min, x_max = dimensions["x"]
    y_min, y_max = dimensions["y"]
    length_x = x_max - x_min
    length_y = y_max - y_min
    dx = length_x / nx
    dy = length_y / ny

    aux_data = {
        "nx": nx,
        "ny": ny,
        "x_bounds": (x_min, x_max),
        "y_bounds": (y_min, y_max),
        "dx": dx,
        "dy": dy,
        "electrodes": electrode_info,
        **phyiscs_data,
    }

    # The geometry containing all the electrodes correctly
    # Everywhere NAN except where there is an electrode
    geometry = jnp.full((nx, ny), jnp.nan)

    # Groups of geometries per electrodes
    # Everywhere NAN except where the is an electrode. The value is 0 in all electrodes except the active one
    anodes = electrode_info.get("anodes", [])
    drifts = electrode_info.get("drifts", [])
    cathodes = electrode_info.get("cathodes", [])
    anodes_geometries = jnp.full((len(anodes), nx, ny), jnp.nan)
    drifts_geometries = jnp.full((len(drifts), nx, ny), jnp.nan)
    cathodes_geometries = jnp.full((len(cathodes), nx, ny), jnp.nan)

    def position_to_pixel(pos: float, dimension: str) -> int:
        """transform from a position in meters to the corresponding pixel

        Args:
            pos (float): position in meters
            dimension (str): whether it is in the `x` or in `y`

        Returns:
            int: corresponding pixel
        """
        if dimension == "x":
            return round((pos - x_min) / dx)
        elif dimension == "y":
            return round((pos - y_min) / dy)

    def set_all_electrodes(
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        anodes_geometries: jnp.ndarray,
        drifts_geometries: jnp.ndarray,
        cathodes_geometries: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Set all the values in all geometries to 0 (necessary step for calculating the potentials)

        Args:
            x_min (int): start x pixel
            x_max (int): end x pixel
            y_min (int): start y pixel
            y_max (int): end y pixel
            anodes_geometries (jnp.ndarray): anodes geometries to set to 0
            drifts_geometries (jnp.ndarray): drifts geometries to set to 0
            cathodes_geometries (jnp.ndarray): cathodes geometries to set to 0

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: The updated 3 arrays
        """
        anodes_geometries = anodes_geometries.at[:, x_min:x_max, y_min:y_max].set(0)
        drifts_geometries = drifts_geometries.at[:, x_min:x_max, y_min:y_max].set(0)
        cathodes_geometries = cathodes_geometries.at[:, x_min:x_max, y_min:y_max].set(0)

        return anodes_geometries, drifts_geometries, cathodes_geometries

    # LOAD ANODES
    for i, anode in enumerate(anodes):
        value = anode["value"]
        a_x_min, a_x_max = anode["position"]["x"]
        a_y_min, a_y_max = anode["position"]["y"]
        x_min_idx = position_to_pixel(a_x_min, "x")
        x_max_idx = position_to_pixel(a_x_max, "x")
        y_min_idx = position_to_pixel(a_y_min, "y")
        # NOTE: Hardcoded that we know that the anodes are at the bottom, so +1 in the ymax
        y_max_idx = position_to_pixel(a_y_max, "y") + 1

        geometry = geometry.at[x_min_idx:x_max_idx, y_min_idx:y_max_idx].set(value)

        # Set all other geometries to 0
        anodes_geometries, drifts_geometries, cathodes_geometries = set_all_electrodes(
            x_min_idx,
            x_max_idx,
            y_min_idx,
            y_max_idx,
            anodes_geometries,
            drifts_geometries,
            cathodes_geometries,
        )

        # Set the correct geometry to 1
        anodes_geometries = anodes_geometries.at[
            i, x_min_idx:x_max_idx, y_min_idx:y_max_idx
        ].set(1)

    # LOAD DRIFTS
    for i, drift in enumerate(drifts):
        value = drift["value"]
        d_x_min0, d_x_max0, d_x_min1, d_x_max1, d_x_min2, d_x_max2 = drift["position"][
            "x"
        ]
        d_y_min0, d_y_max0, d_y_min1, d_y_max1, d_y_min2, d_y_max2 = drift["position"][
            "y"
        ]
        x_min_idx0 = position_to_pixel(d_x_min0, "x")
        x_min_idx1 = position_to_pixel(d_x_min1, "x")
        x_min_idx2 = position_to_pixel(d_x_min2, "x")

        x_max_idx0 = position_to_pixel(d_x_max0, "x")
        x_max_idx1 = position_to_pixel(d_x_max1, "x")
        x_max_idx2 = position_to_pixel(d_x_max2, "x")

        y_min_idx0 = position_to_pixel(d_y_min0, "y")
        y_min_idx1 = position_to_pixel(d_y_min1, "y")
        y_min_idx2 = position_to_pixel(d_y_min2, "y")

        # NOTE: Hardcoded that we know that the drifts are at the bottom, so +1 in the ymax
        y_max_idx0 = position_to_pixel(d_y_max0, "y") + 1
        y_max_idx1 = position_to_pixel(d_y_max1, "y") + 1
        y_max_idx2 = position_to_pixel(d_y_max2, "y") + 1

        # The values are not the same in all of them.
        # The center is 2/3 of the value, and each side is 1/3 of the value
        geometry = geometry.at[x_min_idx0:x_max_idx0, y_min_idx0:y_max_idx0].set(
            value / 3
        )
        geometry = geometry.at[x_min_idx1:x_max_idx1, y_min_idx1:y_max_idx1].set(
            (value * 2) / 3
        )
        geometry = geometry.at[x_min_idx2:x_max_idx2, y_min_idx2:y_max_idx2].set(
            value / 3
        )

        # Set all other geometries to 0
        anodes_geometries, drifts_geometries, cathodes_geometries = set_all_electrodes(
            x_min_idx0,
            x_max_idx0,
            y_min_idx0,
            y_max_idx0,
            anodes_geometries,
            drifts_geometries,
            cathodes_geometries,
        )
        anodes_geometries, drifts_geometries, cathodes_geometries = set_all_electrodes(
            x_min_idx1,
            x_max_idx1,
            y_min_idx1,
            y_max_idx1,
            anodes_geometries,
            drifts_geometries,
            cathodes_geometries,
        )
        anodes_geometries, drifts_geometries, cathodes_geometries = set_all_electrodes(
            x_min_idx2,
            x_max_idx2,
            y_min_idx2,
            y_max_idx2,
            anodes_geometries,
            drifts_geometries,
            cathodes_geometries,
        )

        # Set the correct geometry to 1
        drifts_geometries = drifts_geometries.at[
            i, x_min_idx0:x_max_idx0, y_min_idx0:y_max_idx0
        ].set(1)
        drifts_geometries = drifts_geometries.at[
            i, x_min_idx1:x_max_idx1, y_min_idx1:y_max_idx1
        ].set(1)
        drifts_geometries = drifts_geometries.at[
            i, x_min_idx2:x_max_idx2, y_min_idx2:y_max_idx2
        ].set(1)

    # LOAD CATHODES
    for i, cathode in enumerate(cathodes):
        value = cathode["value"]
        c_x_min, c_x_max = cathode["position"]["x"]
        c_y_min, c_y_max = cathode["position"]["y"]
        x_min_idx = position_to_pixel(c_x_min, "x")
        x_max_idx = position_to_pixel(c_x_max, "x")
        # NOTE: Hardcoded that we know that the cathodes are at the top, so -1 in the ymin
        y_min_idx = position_to_pixel(c_y_min, "y") - 1
        y_max_idx = position_to_pixel(c_y_max, "y")

        geometry = geometry.at[x_min_idx:x_max_idx, y_min_idx:y_max_idx].set(value)

        # Set all other geometries to 0
        anodes_geometries, drifts_geometries, cathodes_geometries = set_all_electrodes(
            x_min_idx,
            x_max_idx,
            y_min_idx,
            y_max_idx,
            anodes_geometries,
            drifts_geometries,
            cathodes_geometries,
        )

        # Set the correct geometry to 1
        cathodes_geometries = cathodes_geometries.at[
            i, x_min_idx:x_max_idx, y_min_idx:y_max_idx
        ].set(1)

    return (
        jnp.array([geometry]),
        anodes_geometries,
        drifts_geometries,
        cathodes_geometries,
        aux_data,
    )
