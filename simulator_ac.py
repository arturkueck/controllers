import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk

# === Physical parameters ===
g = 9.81
L = 1.0
m = 1.0
I = m * L**2
theta0 = np.pi / 4
omega0 = 0.0
dt = 0.05

# === Adaptive Controller ===
class AdaptiveController:
    def __init__(self, gamma=1.0, torque_limit=50):
        self.gamma = gamma
        self.a1_hat = 0.0
        self.a2_hat = 0.0
        self.setpoint = 0.0
        self.torque_limit = torque_limit

    def reset(self):
        self.a1_hat = 0.0
        self.a2_hat = 0.0

    def get_torque(self, error, omega):
        return self.a1_hat * error + self.a2_hat * omega

    def update(self, error, omega, dt):
        self.a1_hat += self.gamma * error * error * dt
        self.a2_hat += self.gamma * error * omega * dt

        self.a1_hat = np.clip(self.a1_hat, -self.torque_limit, self.torque_limit)
        self.a2_hat = np.clip(self.a2_hat, -self.torque_limit, self.torque_limit)

controller = AdaptiveController(gamma=0.5)

# === State and logs ===
state = [theta0, omega0]
time = [0.0]
time_history = []
torque_history = []
is_paused = False
debug_shown = False

# === Plot setup ===
fig, (ax_pendulum, ax_torque) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.4})
plt.subplots_adjust(left=0.15, bottom=0.4)

# -- Pendulum
ax_pendulum.set_xlim(-L - 0.2, L + 0.2)
ax_pendulum.set_ylim(-L - 0.2, L + 0.2)
ax_pendulum.set_aspect('equal')
pendulum_line, = ax_pendulum.plot([], [], 'o-', lw=2)
pendulum_title = ax_pendulum.set_title("Adaptive-Controlled Pendulum")

# -- Torque plot
ax_torque.set_xlim(0, 10)
ax_torque.set_ylim(-60, 60)
ax_torque.set_title("Torque over Time")
ax_torque.set_xlabel("Time [s]")
ax_torque.set_ylabel("Torque [Nm]")
torque_line, = ax_torque.plot([], [], lw=2, color='tab:red')

# === Sliders ===
gamma_slider_ax = plt.axes([0.25, 0.30, 0.5, 0.025])
angle_slider_ax = plt.axes([0.25, 0.25, 0.5, 0.025])
setpoint_slider_ax = plt.axes([0.25, 0.20, 0.5, 0.025])
gamma_slider = Slider(gamma_slider_ax, 'γ (Adapt Speed)', 0, 0.5, valinit=0.1)
angle_slider = Slider(angle_slider_ax, 'Push Angle (°)', 0.0, 30.0, valinit=10.0)
setpoint_slider = Slider(setpoint_slider_ax, 'Setpoint (°)', -180.0, 180.0, valinit=0.0)

def update_gamma(val):
    controller.gamma = gamma_slider.val
gamma_slider.on_changed(update_gamma)

def update_setpoint(val):
    controller.setpoint = np.radians(setpoint_slider.val)
setpoint_slider.on_changed(update_setpoint)

def reset_simulation(event):
    state[0] = controller.setpoint  # θ to setpoint
    state[1] = 0.0                  # ω to 0
    controller.reset()             # reset adaptive parameters
    time[0] = 0.0
    time_history.clear()
    torque_history.clear()


# === Buttons ===
push_left_ax = plt.axes([0.1, 0.12, 0.15, 0.05])
pause_ax = plt.axes([0.42, 0.12, 0.15, 0.05])
push_right_ax = plt.axes([0.75, 0.12, 0.15, 0.05])
button_left = Button(push_left_ax, 'Push Left', color='lightgray')
pause_button = Button(pause_ax, 'Pause', color='lightgray')
button_right = Button(push_right_ax, 'Push Right', color='lightgray')

reset_ax = plt.axes([0.42, 0.05, 0.15, 0.05])
reset_button = Button(reset_ax, 'Reset', color='lightgray')
reset_button.on_clicked(reset_simulation)



def push_left(event): state[0] -= np.radians(angle_slider.val)
def push_right(event): state[0] += np.radians(angle_slider.val)

def toggle_pause(event):
    global is_paused, debug_shown
    is_paused = not is_paused
    pause_button.label.set_text("Resume" if is_paused else "Pause")
    if is_paused:
        ani.event_source.stop()
        if not debug_shown:
            show_debug()
            debug_shown = True
    else:
        ani.event_source.start()
        debug_shown = False


button_left.on_clicked(push_left)
button_right.on_clicked(push_right)
pause_button.on_clicked(toggle_pause)

# === Debug Info Window ===
def show_debug():
    theta, omega = state
    setpoint = controller.setpoint
    error = setpoint - theta
    derr = omega  # since d(error)/dt = -omega
    torque = controller.get_torque(error, omega)
    u_total = torque + (g / L) * np.sin(theta) * I

    fig_d, ax_d = plt.subplots(figsize=(6, 5))
    fig_d.canvas.manager.set_window_title("Adaptive Control Equation Breakdown")
    ax_d.axis("off")

    # LaTeX equation (control law)
    latex = (
        r"$u(t) = \hat{a}_1 \cdot e(t) + \hat{a}_2 \cdot \omega + \frac{g}{L} \sin(\theta)$"
    )

    # Monospaced debug breakdown
    txt = (
        f"1. θ (actual angle)        = {np.degrees(theta): .2f}°\n"
        f"2. ω (angular velocity)    = {omega: .3f} rad/s\n"
        f"3. θd (desired angle)      = {np.degrees(setpoint): .1f}°\n\n"

        f"4. e(t) = θd - θ           = {np.degrees(error): .2f}°\n"
        f"5. de(t)/dt = -ω          = {-omega: .3f} rad/s\n\n"

        f"6. u_partial = a1_hat·e + a2_hat·ω\n"
        f"   = {controller.a1_hat:.2f}×{np.degrees(error):.2f} + {controller.a2_hat:.2f}×({omega:.2f})\n"
        f"   = {torque:.3f} Nm (adaptive torque)\n\n"

        f"7. u_total = u_partial + (g/L)·sin(θ)\n"
        f"   = {torque:.3f} + {(g/L):.2f}×sin({np.degrees(theta):.2f}°)\n"
        f"   = {u_total:.3f} Nm (actual torque)"
    )

    
    latex2 = (
        r"$\dot{\hat{a}}_1 = \gamma \cdot e^2, \quad \dot{\hat{a}}_2 = \gamma \cdot e \cdot \omega$"
    )
    ax_d.text(0.1, 0.85, latex, ha="left", va="top", fontsize=14)
    ax_d.text(0.1, 0.75, latex2, ha="left", va="top", fontsize=13)
    
    
    # Plot the equation and explanation
    ax_d.text(0.1, 0.55, txt, ha="left", va="top", family="monospace", fontsize=11)
    
    plt.tight_layout()
    plt.show()


# === ODE system ===
def pendulum_ode(t, y):
    theta, omega = y
    error = controller.setpoint - theta
    torque = controller.get_torque(error, omega) + (g / L) * np.sin(theta) * I
    torque = np.clip(torque, -controller.torque_limit, controller.torque_limit)
    controller.update(error, omega, dt)
    return [omega, - (g / L) * np.sin(theta) + torque / I]


# === Animation ===
def animate(frame):
    global state, time, debug_shown

    if not is_paused:
        sol = solve_ivp(pendulum_ode, [0, dt], state, t_eval=[dt])
        if sol.success:
            state[:] = sol.y[:, -1]
            time[0] += dt
        debug_shown = False

    error = np.clip(controller.setpoint - state[0], -np.pi, np.pi)
    torque = controller.get_torque(error, state[1]) + (g / L) * np.sin(state[0]) * I
    torque = np.clip(torque, -controller.torque_limit, controller.torque_limit)

    x = L * np.sin(state[0])
    y = -L * np.cos(state[0])
    pendulum_line.set_data([0, x], [0, y])
    pendulum_title.set_text(f"t={time[0]:.1f}s | θ={np.degrees(state[0]):+.1f}° | τ={torque:.2f} Nm")

    time_history.append(time[0])
    torque_history.append(torque)
    while time_history and time[0] - time_history[0] > 10:
        time_history.pop(0)
        torque_history.pop(0)
    ax_torque.set_xlim(max(0, time[0]-10), time[0])
    torque_line.set_data(time_history, torque_history)
    ax_torque.relim()
    ax_torque.autoscale_view(scalex=False, scaley=True)

    return pendulum_line, torque_line

ani = FuncAnimation(fig, animate, interval=50)
plt.show()
