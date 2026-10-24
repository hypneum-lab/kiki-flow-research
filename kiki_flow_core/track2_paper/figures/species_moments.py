"""Figure 1 (v2): per-species mean position and entropy over slow steps.

Replaces the former phase portrait, which plotted ``rho.mean()`` -- a
constant ``1/GRID`` for any normalised histogram -- and so showed a single
point whatever the dynamics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SPECIES = ["phono", "lex", "syntax", "sem"]


def species_moments(rho: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean position and Shannon entropy (bits) of histograms.

    ``rho`` has shape ``(..., grid)`` and is normalised on the last axis;
    ``x`` is the grid (shape ``(grid,)``). Returns two arrays of shape ``(...)``.
    """
    mean = rho @ x
    safe = np.clip(rho, 1e-12, None)
    entropy = -(safe * np.log2(safe)).sum(axis=-1)
    return mean, entropy


def make_species_moments(
    runs: dict[str, np.ndarray],
    x: np.ndarray,
    out_dir: Path,
    filename: str = "fig1_species_moments",
) -> Path:
    """Plot mean position and entropy per species over slow steps.

    ``runs`` maps a label (e.g. "fast", "rigorous") to an array of shape
    ``(n_seeds, n_steps, n_species, grid)``. Lines show the seed mean, the
    band +/- one seed standard deviation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = ["-", "--", ":", "-."]
    for r_idx, (label, arr) in enumerate(runs.items()):
        mean, entropy = species_moments(arr, x)
        steps = np.arange(1, arr.shape[1] + 1)
        for s_idx, name in enumerate(SPECIES[: arr.shape[2]]):
            for ax, val in ((axes[0], mean), (axes[1], entropy)):
                mu = val[:, :, s_idx].mean(axis=0)
                sd = val[:, :, s_idx].std(axis=0)
                ax.plot(
                    steps,
                    mu,
                    color=colors[s_idx % len(colors)],
                    linestyle=styles[r_idx % len(styles)],
                    linewidth=1.0,
                    label=f"{name} ({label})" if ax is axes[0] else None,
                )
                ax.fill_between(steps, mu - sd, mu + sd, color=colors[s_idx], alpha=0.12)
    axes[0].set_ylabel("mean position  E[x]")
    axes[1].set_ylabel("entropy (bits)")
    axes[1].set_xlabel("slow step")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].set_title("T2 per-species moments (mean +/- sd over seeds)")
    fig.tight_layout()
    png_path = out_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path
