from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ============================================================
# Parameters
# ============================================================


@dataclass
class VicsekParameters:
    n_particles: int = 300
    box_size: float = 30.0
    interaction_radius: float = 1.0
    speed: float = 0.03
    noise_amplitude: float = 2.0
    dt: float = 1.0
    seed: int = 42

    @property
    def density(self) -> float:
        return self.n_particles / self.box_size**2


# ============================================================
# Simulation
# ============================================================


def order_parameter(theta: np.ndarray) -> float:
    return float(np.hypot(np.mean(np.cos(theta)), np.mean(np.sin(theta))))


def simulate_vicsek(
    params: VicsekParameters,
    n_steps: int = 700,
    save_every: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(params.seed)

    n_saved = n_steps // save_every

    positions_history = np.empty((n_saved, params.n_particles, 2), dtype=float)
    theta_history = np.empty((n_saved, params.n_particles), dtype=float)
    order_history = np.empty(n_saved, dtype=float)

    positions = rng.uniform(
        0.0,
        params.box_size,
        size=(params.n_particles, 2),
    )

    theta = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=params.n_particles,
    )

    r2 = params.interaction_radius**2
    L = params.box_size

    saved_index = 0

    for step in range(n_steps):
        x = positions[:, 0]
        y = positions[:, 1]

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]

        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)

        neighbors = dx * dx + dy * dy <= r2

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        mean_sin = neighbors @ sin_theta
        mean_cos = neighbors @ cos_theta

        mean_theta = np.arctan2(mean_sin, mean_cos)

        noise = rng.uniform(
            -params.noise_amplitude / 2.0,
            params.noise_amplitude / 2.0,
            size=params.n_particles,
        )

        theta = mean_theta + noise

        positions[:, 0] += params.speed * params.dt * np.cos(theta)
        positions[:, 1] += params.speed * params.dt * np.sin(theta)
        positions %= L

        if step % save_every == 0:
            positions_history[saved_index] = positions
            theta_history[saved_index] = theta
            order_history[saved_index] = order_parameter(theta)
            saved_index += 1

    return positions_history, theta_history, order_history


# ============================================================
# Animation
# ============================================================


def add_orientation_color_wheel(fig, ax, cmap: str = "hsv", inner_radius: float = 0.40):
    bbox = ax.get_position()
    wheel_left = bbox.x1 + 0.035
    wheel_size = min(
        min(bbox.width, bbox.height) * 0.36,
        0.98 - wheel_left,
    )
    wheel_bottom = bbox.y1 - wheel_size

    wheel_ax = fig.add_axes(
        [wheel_left, wheel_bottom, wheel_size, wheel_size],
    )

    grid_size = 256
    x = np.linspace(-1.0, 1.0, grid_size)
    y = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(x, y)

    radius = np.hypot(xx, yy)
    angle = np.arctan2(yy, xx) % (2.0 * np.pi)
    wheel = np.ma.array(angle, mask=(radius > 1.0) | (radius < inner_radius))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)

    wheel_ax.imshow(
        wheel,
        origin="lower",
        extent=(-1.0, 1.0, -1.0, 1.0),
        cmap=cmap_obj,
        vmin=0.0,
        vmax=2.0 * np.pi,
        interpolation="bilinear",
    )

    wheel_ax.set_aspect("equal")
    wheel_ax.set_xlim(-1.25, 1.25)
    wheel_ax.set_ylim(-1.25, 1.25)
    wheel_ax.set_title("orientation", fontsize=12, pad=10)
    wheel_ax.text(1.12, 0.0, "0", ha="left", va="center", fontsize=11)
    wheel_ax.text(0.0, 1.12, r"$\pi/2$", ha="center", va="bottom", fontsize=11)
    wheel_ax.text(-1.12, 0.0, r"$\pi$", ha="right", va="center", fontsize=11)
    wheel_ax.text(0.0, -1.12, r"$3\pi/2$", ha="center", va="top", fontsize=11)
    wheel_ax.set_axis_off()

    return wheel_ax


def animate_vicsek(
    params: VicsekParameters,
    n_steps: int = 700,
    save_every: int = 1,
    interval_ms: int = 20,
    marker_size: float = 25.0,
    arrow_length: float = 0.5,
    figsize: tuple[float, float] = (9.5, 8.0),
):
    positions_history, theta_history, order_history = simulate_vicsek(
        params=params,
        n_steps=n_steps,
        save_every=save_every,
    )

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.74)

    ax.set_xlim(0.0, params.box_size)
    ax.set_ylim(0.0, params.box_size)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    positions0 = positions_history[0]
    theta0 = theta_history[0]

    colors0 = theta0 % (2.0 * np.pi)

    scatter = ax.scatter(
        positions0[:, 0],
        positions0[:, 1],
        c=colors0,
        cmap="hsv",
        vmin=0.0,
        vmax=2.0 * np.pi,
        s=marker_size,
        linewidths=0,
    )

    arrows = ax.quiver(
        positions0[:, 0],
        positions0[:, 1],
        np.cos(theta0),
        np.sin(theta0),
        angles="xy",
        scale_units="xy",
        scale=1.0 / arrow_length,
        width=0.005,
        color="black",
        alpha=0.75,
    )

    wheel_ax = add_orientation_color_wheel(fig, ax)

    title = ax.set_title("")
    frame_state = {
        "index": 0,
        "is_running": True,
    }

    def update(frame_index: int):
        positions = positions_history[frame_index]
        theta = theta_history[frame_index]
        step = frame_index * save_every

        scatter.set_offsets(positions)
        scatter.set_array(theta % (2.0 * np.pi))

        arrows.set_offsets(positions)
        arrows.set_UVC(np.cos(theta), np.sin(theta))

        title.set_text(
            f"Vicsek model | step = {step} | "
            rf"$\phi$ = {order_history[frame_index]:.3f} | "
            rf"$\rho$ = {params.density:.3f}"
        )

        return scatter, arrows, title

    def frame_sequence():
        while True:
            frame_index = frame_state["index"]
            frame_state["index"] = (frame_index + 1) % len(order_history)
            yield frame_index

    def show_frame(frame_index: int):
        update(frame_index)
        frame_state["index"] = (frame_index + 1) % len(order_history)
        fig.canvas.draw_idle()

    anim = FuncAnimation(
        fig,
        update,
        frames=frame_sequence,
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )

    wheel_bbox = wheel_ax.get_position()
    button_gap = 0.012
    button_height = 0.045
    button_bottom = wheel_bbox.y0 - button_height - 0.035
    button_width = 0.5 * (wheel_bbox.width - button_gap)

    play_ax = fig.add_axes(
        [wheel_bbox.x0, button_bottom, button_width, button_height],
    )
    reset_ax = fig.add_axes(
        [
            wheel_bbox.x0 + button_width + button_gap,
            button_bottom,
            button_width,
            button_height,
        ],
    )

    play_button = Button(play_ax, "Pause")
    reset_button = Button(reset_ax, "Reset")

    def toggle_play(_event):
        if frame_state["is_running"]:
            anim.event_source.stop()
            play_button.label.set_text("Play")
        else:
            anim.event_source.start()
            play_button.label.set_text("Pause")

        frame_state["is_running"] = not frame_state["is_running"]
        fig.canvas.draw_idle()

    def reset_animation(_event):
        show_frame(0)

    play_button.on_clicked(toggle_play)
    reset_button.on_clicked(reset_animation)

    fig._vicsek_buttons = (play_button, reset_button)

    return fig, anim


# ============================================================
# Run locally
# ============================================================

if __name__ == "__main__":

    # #  Low density, high noise: $L=30$, $\eta=2$
    # L = 30.0
    # eta = 2.0

    #  Small density and noise: $L=30$, $\eta=0.1$
    L = 30.0
    eta = 0.1

    # #  High Density, low Noise: $L=5$, $\eta=0.1$
    # L = 5.0
    # eta = 0.1

    params = VicsekParameters(
        n_particles=300,
        box_size=L,
        interaction_radius=1.0,
        speed=0.03,
        noise_amplitude=eta,
        dt=1.0,
        seed=42,
    )

    fig, anim = animate_vicsek(
        params=params,
        n_steps=700,
        save_every=1,
        interval_ms=15,
        marker_size=80.0,
        arrow_length=0.8,
        figsize=(10.5, 8.5),
    )

    plt.show()
