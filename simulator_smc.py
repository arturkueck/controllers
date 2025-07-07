import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')

# === Physical parameters ===
g = 9.81
L = 1.0
m = 1.0
I = m * L**2
dt = 0.05

# === Initial SMC Parameters ===
lambda_init = 5.0
eta_init = 30
epsilon_init = 0.75

# === State and disturbance ===
theta0 = 0.2  # pendulum pointing down
omega0 = 0.0
state = [theta0, omega0]
time = [0.0]
disturbance_torque = [0.0]
torque_history = []
time_history = []

is_paused = False

# === Plot setup ===
fig, (ax_pendulum, ax_torque) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1]})
plt.subplots_adjust(bottom=0.45, hspace=0.4)

# -- Pendulum
ax_pendulum.set_xlim(-L - 0.2, L + 0.2)
ax_pendulum.set_ylim(-L - 0.2, L + 0.2)
ax_pendulum.set_aspect('equal')
line, = ax_pendulum.plot([], [], 'o-', lw=2)
title = ax_pendulum.set_title("Sliding Mode Controlled Pendulum")

# -- Torque Plot
ax_torque.set_xlim(0, 10)
ax_torque.set_ylim(-50, 50)
ax_torque.set_xlabel("Time [s]")
ax_torque.set_ylabel("Torque [Nm]")
torque_line, = ax_torque.plot([], [], lw=2, color='tab:red')

# === Sliders ===
ax_lambda = plt.axes([0.25, 0.35, 0.6, 0.03])
ax_eta = plt.axes([0.25, 0.30, 0.6, 0.03])
ax_eps = plt.axes([0.25, 0.25, 0.6, 0.03])

slider_lambda = Slider(ax_lambda, 'λ (lambda)', 0, 20.0, valinit=lambda_init)
slider_eta = Slider(ax_eta, 'η (eta)', 0, 100.0, valinit=eta_init)
slider_eps = Slider(ax_eps, 'ε (epsilon)', 0.001, 1, valinit=epsilon_init)

ax_lmodel = plt.axes([0.25, 0.20, 0.6, 0.03])
ax_gmodel = plt.axes([0.25, 0.15, 0.6, 0.03])

slider_lmodel = Slider(ax_lmodel, 'L_model', 0.5, 2.0, valinit=L)
slider_gmodel = Slider(ax_gmodel, 'g_model', 0.0, 50.0, valinit=g)

ax_theta_d = plt.axes([0.25, 0.10, 0.6, 0.03])
slider_theta_d = Slider(ax_theta_d, 'setpoint (°)', -180, 180, valinit=0.0)

# Inside smc_control():
theta_d = np.radians(slider_theta_d.val)


# === Push Buttons ===
push_left_ax = plt.axes([0.1, 0.05, 0.15, 0.05])
push_right_ax = plt.axes([0.75, 0.05, 0.15, 0.05])

button_left = Button(push_left_ax, 'Push Left', color='lightgray', hovercolor='gray')
button_right = Button(push_right_ax, 'Push Right', color='lightgray', hovercolor='gray')

def push_left(event):
    disturbance_torque[0] += 20.0

def push_right(event):
    disturbance_torque[0] -= 20.0

button_left.on_clicked(push_left)
button_right.on_clicked(push_right)

pause_ax = plt.axes([0.42, 0.05, 0.15, 0.05])  # center bottom
pause_button = Button(pause_ax, 'Pause', color='lightgray', hovercolor='gray')

def toggle_pause(event):
    global is_paused, debug_shown
    is_paused = not is_paused
    debug_shown = False  # reset when toggled
    pause_button.label.set_text('Resume' if is_paused else 'Pause')


pause_button.on_clicked(toggle_pause)


# === SMC control law ===
def smc_control(theta, omega):
    lam = slider_lambda.val
    eta = slider_eta.val
    eps = slider_eps.val

    L_model = slider_lmodel.val
    g_model = slider_gmodel.val
    theta_d = wrap_angle(np.radians(slider_theta_d.val))

    s = omega + lam * (theta - theta_d)
    sat = np.tanh(s / eps)

    u = -I * (g_model / L_model * np.sin(theta) + lam * omega) - eta * sat
    return u, s, sat, theta_d


def update_debug_window(theta, omega, theta_d, lam, eta, eps, u, s, sat):
    fig_debug, ax_debug = plt.subplots(figsize=(7, 6))
    fig_debug.canvas.manager.set_window_title("SMC Equation Breakdown")

    ax_debug.axis("off")

    latex_formula = (
        r"$u = -I \left( \frac{g_\mathrm{model}}{L_\mathrm{model}} \sin(\theta) + "
        r"\lambda \cdot \omega \right) - \eta \cdot \tanh\left( \frac{s}{\varepsilon} \right)$"
        "\n\n"
        r"$s = \omega + \lambda(\theta - \theta_d)$"
    )

    text_info = (
        f"θ = {np.degrees(theta):.2f}° ({theta:.4f} rad)\n"
        f"setpoint = {np.degrees(theta_d):.2f}° ({theta_d:.4f} rad)\n"
        f"ω = {omega:.4f} rad/s\n"
        f"λ = {lam:.2f}, η = {eta:.2f}, ε = {eps:.3f}\n\n"
        f"s = {s:.4f}\n"
        f"sat(s/ε) = {sat:.4f}\n\n"
        f"u = {u:.4f} Nm"
    )

    ax_debug.text(0.05, 0.85, latex_formula, ha='left', va='top', fontsize=14)
    ax_debug.text(0.05, 0.6, text_info, ha='left', va='top', fontsize=11, family='monospace')
    plt.tight_layout()
    plt.show()


# === Animation ===
def animate(frame):
    global state, time, debug_shown
    
    if is_paused:
        if not debug_shown:
            lam = slider_lambda.val
            eta = slider_eta.val
            eps = slider_eps.val
            theta_d = wrap_angle(np.radians(slider_theta_d.val))  # Add this line
            theta, omega = state
            s = omega + lam * (theta - theta_d)
            sat_val = np.tanh(s / eps)
            u = -I * (slider_gmodel.val / slider_lmodel.val * np.sin(theta) + lam * omega) - eta * sat_val
            update_debug_window(theta, omega, theta_d, lam, eta, eps, u, s, sat_val)
            debug_shown = True
        return line, torque_line

    theta, omega = state
    u, s, sat_val, theta_d = smc_control(theta, omega)
    u += disturbance_torque[0]
    
    MAX_TORQUE = 50.0
    u = np.clip(u, -MAX_TORQUE, MAX_TORQUE)
    
    disturbance_torque[0] *= 0.9

    # Save for torque plot
    torque_history.append(u)
    time_history.append(time[0])

    # Trim plot history
    while time_history and time[0] - time_history[0] > 4:
        time_history.pop(0)
        torque_history.pop(0)

    def ode(t, y):
        theta, omega = y
        dtheta_dt = omega
        domega_dt = - (g / L) * np.sin(theta) + u / I
        return [dtheta_dt, domega_dt]

    sol = solve_ivp(ode, [0, dt], state, t_eval=[dt])
    theta, omega = sol.y[:, -1]
    state[:] = [theta, omega]
    time[0] += dt

    # Pendulum animation
    x_tip = L * np.sin(theta)
    y_tip = -L * np.cos(theta)
    line.set_data([0, x_tip], [0, y_tip])
    title.set_text(f"t={time[0]:.1f}s | θ={np.degrees(theta):+.1f}° | τ={u:.2f}")

    # Torque plot
    ax_torque.set_xlim(max(0, time[0] - 4), time[0])
    torque_line.set_data(time_history, torque_history)
    ax_torque.relim()
    ax_torque.autoscale_view(scalex=False, scaley=True)

    return line, torque_line

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


ani = FuncAnimation(fig, animate, interval=50)
plt.show()
