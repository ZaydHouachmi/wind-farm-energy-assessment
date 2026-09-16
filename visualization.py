import numpy as np
import matplotlib.pyplot as plt
from wind_farm import WindFarm


def plot_layout(farm: WindFarm, ax=None, show_labels: bool = True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    xs = np.array([t.x for t in farm.turbines])
    ys = np.array([t.y for t in farm.turbines])
    diam = np.array([t.diameter for t in farm.turbines])

    for t in farm.turbines:
        radius = t.diameter / 2.0
        circle = plt.Circle((t.x, t.y), radius=radius,
                             facecolor="steelblue", edgecolor="black",
                             linewidth=1.0, alpha=0.8, zorder=3)
        ax.add_patch(circle)
        if show_labels:
            label_offset = max(12, int(radius * 0.5))
            ax.annotate(t.id, (t.x, t.y), textcoords="offset points",
                        xytext=(0, label_offset),
                        ha="center", va="bottom", fontsize=9, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7))

    ax.scatter(xs, ys, s=10, color="black", zorder=5)

    pad = max(diam.max() * 4, 50) if len(diam) else 50
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_title(f"{farm.name} -- Layout")
    ax.grid(True, linestyle=":", alpha=0.5)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    nx = x1 - 0.08 * (x1 - x0)
    ny = y1 - 0.20 * (y1 - y0)
    arrow_len = 0.08 * (y1 - y0)
    ax.annotate("N", xy=(nx, ny + arrow_len), xytext=(nx, ny),
                arrowprops=dict(facecolor="black", width=2, headwidth=8),
                ha="center", fontsize=11, fontweight="bold")

    if farm.wind_direction is not None:
        d_hat, _ = farm.wind_unit_vectors()
        wx = x0 + 0.12 * (x1 - x0)
        wy = y1 - 0.20 * (y1 - y0)
        arrow_len_w = 0.10 * (y1 - y0)
        ax.annotate(
            "", xy=(wx + d_hat[0] * arrow_len_w, wy + d_hat[1] * arrow_len_w),
            xytext=(wx, wy),
            arrowprops=dict(facecolor="firebrick", edgecolor="firebrick", width=2, headwidth=9),
        )
        ax.text(wx + 0.02 * (x1 - x0), wy - arrow_len_w * 0.7,
                 f"Wind: {farm.wind_direction:.0f}°  ({farm.wind_speed:.1f} m/s)",
                 color="firebrick", fontsize=9, ha="left", va="top")

    return ax
