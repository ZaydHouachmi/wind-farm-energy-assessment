"""
demo_interactive.py
===================
Interactive demo with sliders showing layout, wake losses, power curve,
speed field and annual production estimate.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button, RadioButtons
import logging
from datetime import datetime

from turbine import Turbine
from wind_farm import WindFarm
from wake_model import JensenWakeModel, EddyViscosityWakeModel
from power_curve import PowerCurve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track last enforced limits to avoid spamming logs
_last_enforced = None

def _on_draw(event):
    global _last_enforced
    try:
        if layout_extents is None:
            return
        cur_xlim = ax_layout.get_xlim()
        cur_ylim = ax_layout.get_ylim()
        desired_xlim = (float(layout_extents[0]), float(layout_extents[1]))
        desired_ylim = (float(layout_extents[2]), float(layout_extents[3]))
        if cur_xlim != desired_xlim or cur_ylim != desired_ylim:
            now = datetime.now()
            # avoid rapid repeated enforcement
            if _last_enforced is None or now is None or (now - _last_enforced).total_seconds() > 0.05:
                pos = ax_layout.get_position()
                fig_size = fig.get_size_inches()
                logger.info(f"Draw event: axes changed unexpectedly: cur_xlim={cur_xlim}, cur_ylim={cur_ylim}; enforcing layout_extents={layout_extents}; ax_pos={pos}, fig_size={fig_size}")
                ax_layout.set_xlim(desired_xlim)
                ax_layout.set_ylim(desired_ylim)
                ax_layout.autoscale(False)
                _last_enforced = now
    except Exception:
        logger.exception("Error in draw_event handler")

# connect draw handler
try:
    fig_canvas = None
    # fig may not yet be defined at import time, so connect after figure creation below
except Exception:
    pass


def create_farm(rows: int, cols: int, spacing: float, diameter: float, hub_height: float) -> WindFarm:
    farm = WindFarm(name=f"Wind Farm {rows}x{cols}")
    turbines = []
    tid = 1
    for row in range(rows):
        for col in range(cols):
            x = col * spacing
            y = row * spacing
            turbines.append(Turbine(id=f"T{tid:02d}", x=x, y=y,
                                     diameter=diameter, hub_height=hub_height))
            tid += 1
    farm.add_turbines(turbines)
    return farm


def compute_speed_field(farm: WindFarm, wind_speed: float, d_hat: np.ndarray,
                        c_hat: np.ndarray, grid_size: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.array([[t.x, t.y] for t in farm.turbines])
    pad = max([t.diameter for t in farm.turbines]) * 3.0
    x_min, x_max = positions[:, 0].min() - pad, positions[:, 0].max() + pad
    y_min, y_max = positions[:, 1].min() - pad, positions[:, 1].max() + pad

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(xs, ys)
    total_deficit = np.zeros_like(X)

    for turbine in farm.turbines:
        delta_x = X - turbine.x
        delta_y = Y - turbine.y
        xi = delta_x * d_hat[0] + delta_y * d_hat[1]
        eta = delta_x * c_hat[0] + delta_y * c_hat[1]
        deficit = wake_model.wake_deficit(xi, eta, turbine.diameter)
        deficit[xi <= 0] = 0.0
        total_deficit += deficit

    total_deficit = np.clip(total_deficit, 0.0, 0.95)
    speed_field = wind_speed * (1.0 - total_deficit)
    return speed_field, X, Y


# Initial farm parameters
D = 90.0
H = 80.0
SPACING = 5 * D

state = {
    'speed': 9.0,
    'direction': 270.0,
    'playing': False,
    'rows': 2,
    'cols': 3,
    'spacing': SPACING,
    'diameter': D,
    'hub_height': H,
    'model': 'Viscosity',
}

farm = create_farm(state['rows'], state['cols'], state['spacing'],
                   state['diameter'], state['hub_height'])

# Persistent layout extents (xmin, xmax, ymin, ymax) to prevent autoscale shrinking
layout_extents = None


wake_models = {
    'Viscosity': EddyViscosityWakeModel(ambient_turbulence=0.10, thrust_coeff=0.8),
    'Jensen': JensenWakeModel(wake_decay=0.05),
}
wake_model = wake_models[state['model']]
power_curve = PowerCurve(cut_in=3.0, rated_speed=11.0, cut_out=25.0,
                        rated_power=2500.0)

# Create figure with 4 subplots (taller for clearer control spacing)
fig = plt.figure(figsize=(16, 11))
ax_layout = fig.add_subplot(2, 2, 1)
ax_losses = fig.add_subplot(2, 2, 2)
ax_power = fig.add_subplot(2, 2, 3)
ax_curve = fig.add_subplot(2, 2, 4)

# connect draw handler now figure exists
try:
    fig.canvas.mpl_connect('draw_event', _on_draw)
    def _on_resize(event):
        try:
            pos = ax_layout.get_position()
            fig_size = fig.get_size_inches()
            logger.info(f"Resize event: fig_size={fig_size}, ax_layout_pos={pos}")
        except Exception:
            logger.exception('Error in resize handler')

    fig.canvas.mpl_connect('resize_event', _on_resize)
except Exception:
    logger.exception('Could not connect draw_event handler')

plt.subplots_adjust(left=0.10, right=0.99, top=0.83, bottom=0.345, hspace=0.58, wspace=0.35)
fig.patch.set_facecolor('white')

# Helper to create a composite slider control with a dedicated label axis.
def create_slider_control(fig, x, y, width, height, title, vmin, vmax, valinit,
                          value_formatter, valstep=1.0, color='darkgreen', label_fraction=0.30):
    label_width = width * label_fraction
    slider_width = width - label_width

    label_ax = fig.add_axes([x, y, label_width, height])
    label_ax.set_axis_off()
    label_text = label_ax.text(1.0, 0.5, f'{title}: {value_formatter(valinit)}',
                               ha='right', va='center', fontsize=10)

    slider_ax = fig.add_axes([x + label_width, y, slider_width, height])
    slider = Slider(slider_ax, '', vmin, vmax, valinit=valinit, valstep=valstep, color=color)
    slider.label.set_visible(False)
    slider.valtext.set_visible(False)
    return slider, label_text

# Control axes arranged in 3 rows with programmatic layout
# Row 1: Speed and Direction (full-width span starting at x=0.10), stacked
ax_speed = fig.add_axes([0.10, 0.235, 0.50, 0.035])
ax_direction = fig.add_axes([0.10, 0.195, 0.50, 0.035])

# Row A: Rows, Cols, Rotor dia
row_a_x = 0.08
row_a_y = 0.155
row_a_width = 0.72
row_a_gap = 0.03
row_a_control_width = (row_a_width - 2 * row_a_gap) / 3
row_a_positions = [row_a_x + i * (row_a_control_width + row_a_gap) for i in range(3)]

slider_rows, label_rows = create_slider_control(
    fig, row_a_positions[0], row_a_y, row_a_control_width, 0.035,
    'Rows', 1, 5, state['rows'], lambda v: f'{int(round(v))}',
    valstep=1, color='darkgreen')
slider_cols, label_cols = create_slider_control(
    fig, row_a_positions[1], row_a_y, row_a_control_width, 0.035,
    'Cols', 1, 6, state['cols'], lambda v: f'{int(round(v))}',
    valstep=1, color='darkgreen')
slider_diameter, label_diameter = create_slider_control(
    fig, row_a_positions[2], row_a_y, row_a_control_width, 0.035,
    'Rotor dia', 60.0, 120.0, state['diameter'], lambda v: f'{int(round(v))} m',
    valstep=1.0, color='navy')

# Row B: Spacing, Hub height
row_b_x = 0.08
row_b_y = 0.110
row_b_width = 0.72
row_b_gap = 0.05
row_b_control_width = (row_b_width - row_b_gap) / 2
row_b_positions = [row_b_x + i * (row_b_control_width + row_b_gap) for i in range(2)]

slider_spacing, label_spacing = create_slider_control(
    fig, row_b_positions[0], row_b_y, row_b_control_width, 0.035,
    'Spacing', 100.0, 700.0, state['spacing'], lambda v: f'{int(round(v))} m',
    valstep=10.0, color='sienna')
slider_hub, label_hub = create_slider_control(
    fig, row_b_positions[1], row_b_y, row_b_control_width, 0.035,
    'Hub height', 60.0, 140.0, state['hub_height'], lambda v: f'{int(round(v))} m',
    valstep=1.0, color='purple')

# Row 3: Model selector and Play button moved lower to clear new slider row
ax_model = fig.add_axes([0.10, 0.02, 0.40, 0.06])
ax_play = fig.add_axes([0.52, 0.02, 0.12, 0.06])

slider_speed = Slider(ax_speed, 'Wind Speed [m/s]', 3.0, 30.0,
                      valinit=state['speed'], color='steelblue')
slider_direction = Slider(ax_direction, 'Wind Direction [deg]', 0.0, 360.0,
                          valinit=state['direction'], color='firebrick')
button_play = Button(ax_play, 'Play')
model_selector = RadioButtons(ax_model, ('Viscosity', 'Jensen'), active=0)

summary_text = fig.text(0.02, 0.94, '', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9, edgecolor='gray'))

speed_colorbar = None
speed_mesh = None

def reset_bar_charts() -> None:
    global bar_losses, bar_power, loss_texts, power_texts
    ids = [t.id for t in farm.turbines]
    x_positions = np.arange(len(ids))

    ax_losses.clear()
    bar_losses = ax_losses.bar(x_positions, [0.0] * len(ids), color='green', alpha=0.7, edgecolor='black')
    loss_texts = [ax_losses.text(i, 0.2, '0.0%', ha='center', fontsize=9) for i in x_positions]
    ax_losses.set_ylabel('Speed Loss [%]', fontweight='bold', labelpad=12)
    ax_losses.set_title('Wake Loss per Turbine', fontsize=11, fontweight='bold')
    ax_losses.grid(axis='y', alpha=0.3)
    ax_losses.set_ylim([0, 5])
    ax_losses.set_xticks(x_positions)
    ax_losses.set_xticklabels(ids, rotation=25, ha='right')

    ax_power.clear()
    bar_power = ax_power.bar(x_positions, [power_curve.power(state['speed'])] * len(ids),
                             color='steelblue', alpha=0.7, edgecolor='black')
    power_texts = [ax_power.text(i, power_curve.power(state['speed']) + 50,
                                   f'{power_curve.power(state["speed"]):.0f}', ha='center', fontsize=9)
                   for i in x_positions]
    ax_power.set_ylabel('Power [kW]', fontweight='bold', labelpad=12)
    ax_power.set_title('Power per Turbine', fontsize=11, fontweight='bold', pad=12)
    ax_power.grid(axis='y', alpha=0.3)
    ax_power.axhline(y=power_curve.rated_power, color='red', linestyle='--', linewidth=1,
                     label='Rated Power')
    ax_power.legend(fontsize=9)
    ax_power.set_ylim([0, power_curve.rated_power * 1.1])
    ax_power.set_xticks(x_positions)
    ax_power.set_xticklabels(ids, rotation=25, ha='right')

wind_speeds = np.linspace(0, 30, 120)
power_output = np.array([power_curve.power(u) for u in wind_speeds])
curve_line, = ax_curve.plot(wind_speeds, power_output, 'b-', linewidth=2.5, label='Power Curve')
curve_point = ax_curve.scatter([state['speed']], [power_curve.power(state['speed'])],
                              color='red', s=100, zorder=5, label='Current Speed')
ax_curve.fill_between(wind_speeds, 0, power_output, alpha=0.25)
ax_curve.axvspan(power_curve.cut_out, 30, color='gray', alpha=0.12)
ax_curve.axvline(power_curve.cut_out, color='darkred', linestyle='--', linewidth=1.5,
                 label='Cut-out speed')
ax_curve.set_xlabel('Wind Speed [m/s]', fontweight='bold')
ax_curve.set_ylabel('Power [kW]', fontweight='bold')
ax_curve.set_title('Power Curve / Annual Energy', fontsize=11, fontweight='bold')
ax_curve.grid(True, alpha=0.3)
ax_curve.set_xlim([0, 30])
ax_curve.set_ylim([0, power_curve.rated_power * 1.1])
ax_curve.legend(fontsize=9, loc='upper left')

ax_energy = ax_curve.twinx()
energy_output = power_output * 8760.0 / 1000.0
energy_line, = ax_energy.plot(wind_speeds, energy_output, 'm--', linewidth=1.8,
                              label='Annual energy (single turbine)')
energy_point = ax_energy.scatter([state['speed']], [power_curve.power(state['speed']) * 8760.0 / 1000.0],
                                color='magenta', s=80, zorder=5)
ax_energy.set_ylabel('Annual energy [MWh]', color='magenta')
ax_energy.set_ylim([0, energy_output.max() * 1.1])
ax_energy.tick_params(axis='y', colors='magenta')


def rebuild_farm() -> None:
    global farm
    farm = create_farm(
        rows=int(round(slider_rows.val)),
        cols=int(round(slider_cols.val)),
        spacing=slider_spacing.val,
        diameter=slider_diameter.val,
        hub_height=slider_hub.val,
    )
    # update deterministic layout extents based on turbine positions and rotor pad
    global layout_extents
    positions = np.array([[t.x, t.y] for t in farm.turbines])
    if positions.size:
        pad = max([t.diameter for t in farm.turbines]) * 3.0
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        layout_extents = (x_min - pad, x_max + pad, y_min - pad, y_max + pad)
    else:
        layout_extents = None
    logger.info(f"Rebuilt farm; layout_extents={layout_extents}")


def compute_wake_and_power() -> tuple[list[dict], float]:
    farm.set_wind_condition(wind_speed=state['speed'], wind_direction=state['direction'])
    xi, eta = farm.wind_aligned_distances()

    results = []
    total_power = 0.0
    for j, turbine in enumerate(farm.turbines):
        deficits = []
        for i, upstream_turbine in enumerate(farm.turbines):
            if xi[i, j] > 0:
                deficits.append(
                    wake_model.wake_deficit(xi[i, j], eta[i, j], upstream_turbine.diameter)
                )

        if deficits:
            u_eff = wake_model.effective_speed(state['speed'], np.array(deficits))
            deficit_total = 1.0 - (u_eff / state['speed'])
        else:
            u_eff = state['speed']
            deficit_total = 0.0

        power = power_curve.power(u_eff)
        total_power += power
        results.append({'id': turbine.id, 'deficit': deficit_total, 'power': power})

    return results, total_power


def draw_layout():
    global speed_colorbar
    ax_layout.clear()

    d_hat, c_hat = farm.wind_unit_vectors()
    # Attempt to draw speed field; fall back to simple layout on error
    try:
        # Use the expensive, higher-resolution grid for the EddyViscosity/Ainslie model only
        grid_size = 40 if isinstance(wake_model, EddyViscosityWakeModel) else 80
        speed_field, X, Y = compute_speed_field(farm, state['speed'], d_hat, c_hat, grid_size=grid_size)

        # Create or update the mesh and reuse a single persistent colorbar to avoid
        # changing axes positions repeatedly (which causes visual shrinking).
        global speed_mesh, speed_colorbar
        if speed_mesh is None:
            speed_mesh = ax_layout.pcolormesh(X, Y, speed_field, cmap='viridis', shading='auto', alpha=0.85)
            if speed_colorbar is None:
                speed_colorbar = fig.colorbar(speed_mesh, ax=ax_layout, pad=0.02)
                speed_colorbar.set_label('Wind speed [m/s]')
        else:
            try:
                speed_mesh.remove()
            except Exception:
                pass
            speed_mesh = ax_layout.pcolormesh(X, Y, speed_field, cmap='viridis', shading='auto', alpha=0.85)
            if speed_colorbar is not None:
                try:
                    speed_colorbar.set_mappable(speed_mesh)
                    speed_colorbar.update_normal(speed_mesh)
                except Exception:
                    # fallback: attempt one-time recreate if update fails
                    try:
                        speed_colorbar.remove()
                        speed_colorbar = fig.colorbar(speed_mesh, ax=ax_layout, pad=0.02)
                        speed_colorbar.set_label('Wind speed [m/s]')
                    except Exception:
                        speed_colorbar = None
        # Apply persistent layout extents if available to avoid autoscale shrinking
        if layout_extents is not None:
            prev_xlim = ax_layout.get_xlim()
            prev_ylim = ax_layout.get_ylim()
            ax_layout.set_xlim(float(layout_extents[0]), float(layout_extents[1]))
            ax_layout.set_ylim(float(layout_extents[2]), float(layout_extents[3]))
            ax_layout.autoscale(False)
            logger.debug(f"draw_layout applied layout_extents, prev_xlim={prev_xlim}, prev_ylim={prev_ylim}, new_xlim={ax_layout.get_xlim()}, new_ylim={ax_layout.get_ylim()}")

        contour = ax_layout.contour(X, Y, speed_field, levels=8, colors='white', linewidths=0.7, alpha=0.7)
        # Safely handle contour collections across Matplotlib versions
        for coll in getattr(contour, 'collections', []):
            try:
                coll.set_linestyle('--')
            except Exception:
                pass
    except Exception:
        # Fallback: no speed field (simpler rendering)
        # Remove existing colorbar if present and replace with a simple static bar so
        # the UI remains stable when the speed-field render fails temporarily.
        try:
            if speed_colorbar is not None:
                speed_colorbar.remove()
        except Exception:
            pass
        try:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=0.0, vmax=max(1.0, state['speed']))
            mappable = ScalarMappable(norm=norm, cmap='viridis')
            mappable.set_array([])
            speed_colorbar = fig.colorbar(mappable, ax=ax_layout, pad=0.02)
            speed_colorbar.set_label('Wind speed [m/s]')
        except Exception:
            speed_colorbar = None
        # Ensure a reasonable axis window around turbines when no speed field
        positions = np.array([[t.x, t.y] for t in farm.turbines])
        if positions.size:
            # If we have precomputed extents, reuse them; otherwise compute a deterministic pad
            if layout_extents is not None:
                ax_layout.set_xlim(float(layout_extents[0]), float(layout_extents[1]))
                ax_layout.set_ylim(float(layout_extents[2]), float(layout_extents[3]))
            else:
                x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
                y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
                pad = max([t.diameter for t in farm.turbines]) * 3.0
                ax_layout.set_xlim(x_min - pad, x_max + pad)
                ax_layout.set_ylim(y_min - pad, y_max + pad)
            ax_layout.autoscale(False)
        ax_layout.set_facecolor('whitesmoke')

    for turbine in farm.turbines:
        radius = turbine.diameter / 2.0
        circle = Circle((turbine.x, turbine.y), radius=radius,
                        facecolor='steelblue', edgecolor='black', alpha=0.85, zorder=5)
        ax_layout.add_patch(circle)
        # larger label offset proportional to rotor radius for better readability
        label_offset = int(round(max(8.0, radius * 0.4)))
        ax_layout.text(turbine.x, turbine.y + label_offset, turbine.id,
                       ha='center', va='bottom', fontsize=10, zorder=6,
                       bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='black', alpha=0.85))

    ax_layout.set_aspect('equal', adjustable='box')
    ax_layout.set_xlabel('Easting [m]', labelpad=10)
    ax_layout.set_ylabel('Northing [m]', labelpad=10)
    ax_layout.grid(True, linestyle=':', alpha=0.4)

    x0, x1 = ax_layout.get_xlim()
    y0, y1 = ax_layout.get_ylim()
    arrow_start = np.array([x0 + 0.08 * (x1 - x0), y0 + 0.08 * (y1 - y0)])
    arrow_len = 0.12 * (y1 - y0)
    end = arrow_start + d_hat * arrow_len
    ax_layout.annotate('', xy=end, xytext=arrow_start,
                       arrowprops=dict(facecolor='firebrick', edgecolor='firebrick', width=2, headwidth=9),
                       zorder=7)
    ax_layout.text(arrow_start[0], arrow_start[1] - 0.05 * (y1 - y0),
                   f'Wind {state["direction"]:.0f}° @ {state["speed"]:.1f} m/s',
                   color='firebrick', fontsize=10, ha='left', va='top', zorder=7,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))

    ax_layout.annotate('N', xy=(x1 - 0.08 * (x1 - x0), y1 - 0.10 * (y1 - y0)),
                       xytext=(x1 - 0.08 * (x1 - x0), y1 - 0.03 * (y1 - y0)),
                       arrowprops=dict(facecolor='black', width=2, headwidth=8),
                       ha='center', fontsize=11, fontweight='bold')

    ax_layout.set_title(
        f"Wind speed field and turbine layout ({farm.name})\n"
        f"{state['rows']}x{state['cols']} grid, D={state['diameter']:.0f} m, s={state['spacing']:.0f} m, "
        f"model={state['model']}",
        fontsize=11, fontweight='bold', pad=14
    )


def update(val):
    state['speed'] = slider_speed.val
    state['direction'] = slider_direction.val
    state['rows'] = int(round(slider_rows.val))
    state['cols'] = int(round(slider_cols.val))
    state['spacing'] = slider_spacing.val
    state['diameter'] = slider_diameter.val
    state['hub_height'] = slider_hub.val
    state['model'] = model_selector.value_selected

    global wake_model
    wake_model = wake_models[state['model']]

    # Update external label text artists to reflect current values.
    try:
        label_rows.set_text(f'Rows: {state["rows"]}')
        label_cols.set_text(f'Cols: {state["cols"]}')
        label_diameter.set_text(f'Rotor dia: {int(round(state["diameter"]))} m')
        label_spacing.set_text(f'Spacing: {int(round(state["spacing"]))} m')
        label_hub.set_text(f'Hub height: {int(round(state["hub_height"]))} m')
    except Exception:
        logger.exception('Error updating slider labels')

    rebuild_farm()
    reset_bar_charts()
    results, total_power = compute_wake_and_power()
    draw_layout()

    # Force persistent layout extents after drawing to prevent any downstream code from
    # changing the axes limits (diagnostic/logging will show any changes).
    if layout_extents is not None:
        prev_xlim = ax_layout.get_xlim()
        prev_ylim = ax_layout.get_ylim()
        ax_layout.set_xlim(float(layout_extents[0]), float(layout_extents[1]))
        ax_layout.set_ylim(float(layout_extents[2]), float(layout_extents[3]))
        ax_layout.autoscale(False)
        if prev_xlim != ax_layout.get_xlim() or prev_ylim != ax_layout.get_ylim():
            logger.info(f"update forced layout_extents: prev_xlim={prev_xlim}, prev_ylim={prev_ylim}, new_xlim={ax_layout.get_xlim()}, new_ylim={ax_layout.get_ylim()}")

    deficits = [r['deficit'] * 100 for r in results]
    max_loss = max(deficits) if deficits else 5
    average_loss = np.mean(deficits) if deficits else 0.0

    for rect, d, text in zip(bar_losses, deficits, loss_texts):
        rect.set_height(d)
        rect.set_color('red' if d > 0 else 'green')
        text.set_y(d + 0.6)
        text.set_text(f'{d:.1f}%')
    ax_losses.set_ylim([0, max(5, max_loss * 1.2)])

    powers = [r['power'] for r in results]
    for rect, p, text in zip(bar_power, powers, power_texts):
        rect.set_height(p)
        text.set_y(p + 50)
        text.set_text(f'{p:.0f}')
    annual_energy = total_power * 8760.0 / 1000.0
    ax_power.set_title(
        f'Power per Turbine (Farm total: {total_power:.0f} kW, Annual: {annual_energy:.0f} MWh)',
        fontsize=11, fontweight='bold'
    )

    curve_point.set_offsets([[state['speed'], power_curve.power(state['speed'])]])
    energy_point.set_offsets([[state['speed'], power_curve.power(state['speed']) * 8760.0 / 1000.0]])

    summary_text.set_text(
        f'Wind model: {state["model"]}  |  Rows x Cols: {state["rows"]}x{state["cols"]}  |  '
        f'Total power: {total_power:.0f} kW  |  Avg wake loss: {average_loss:.2f}%'
    )

    fig.canvas.draw_idle()


def toggle_play(event):
    state['playing'] = not state['playing']
    button_play.label.set_text('Pause' if state['playing'] else 'Play')


def animate():
    if not state['playing']:
        return
    new_dir = (slider_direction.val + 2) % 360
    slider_direction.set_val(new_dir)


# Connect widgets
slider_speed.on_changed(update)
slider_direction.on_changed(update)
slider_rows.on_changed(update)
slider_cols.on_changed(update)
slider_diameter.on_changed(update)
slider_spacing.on_changed(update)
slider_hub.on_changed(update)
model_selector.on_clicked(lambda label: update(None))
button_play.on_clicked(toggle_play)

# Timer animation
timer = fig.canvas.new_timer(interval=100)
timer.add_callback(animate)
timer.start()

# Initial draw
update(None)
plt.suptitle('Interactive Wind Farm Analysis: speed field, wake, power and annual energy',
             fontsize=14, fontweight='bold', y=0.98)
plt.show()
