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
eta_init = 0.5
epsilon_init = 0.05

# === State and disturbance ===
theta0 = 0.2  # pendulum pointing down
omega0 = 0.0
state = [theta0, omega0]
time = [0.0]
disturbance_torque = [0.0]
torque_history = []
time_history = []

# === Plot setup ===
fig, (ax_pendulum, ax_torque) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1]})
plt.subplots_adjust(bottom=0.35, hspace=0.4)

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
ax_lambda = plt.axes([0.25, 0.25, 0.6, 0.03])
ax_eta = plt.axes([0.25, 0.20, 0.6, 0.03])
ax_eps = plt.axes([0.25, 0.15, 0.6, 0.03])

slider_lambda = Slider(ax_lambda, 'λ (lambda)', 0.1, 20.0, valinit=lambda_init)
slider_eta = Slider(ax_eta, 'η (eta)', 0.1, 2.0, valinit=eta_init)
slider_eps = Slider(ax_eps, 'ε (epsilon)', 0.001, 0.1, valinit=epsilon_init)

# === Push Buttons ===
push_left_ax = plt.axes([0.3, 0.05, 0.15, 0.05])
push_right_ax = plt.axes([0.55, 0.05, 0.15, 0.05])

button_left = Button(push_left_ax, 'Push Left', color='lightgray', hovercolor='gray')
button_right = Button(push_right_ax, 'Push Right', color='lightgray', hovercolor='gray')

def push_left(event):
    disturbance_torque[0] += 20.0

def push_right(event):
    disturbance_torque[0] -= 20.0

button_left.on_clicked(push_left)
button_right.on_clicked(push_right)

# === SMC control law ===
def smc_control(theta, omega):
    lam = slider_lambda.val
    eta = slider_eta.val
    eps = slider_eps.val

    s = omega + lam * theta
    sat = np.tanh(s / eps)  # smoothed sign function

    # SMC torque
    u = -I * (g / L * np.sin(theta) + lam * omega) - eta * sat
    return u

# === Animation ===
def animate(frame):
    global state, time

    theta, omega = state
    u = smc_control(theta, omega) + disturbance_torque[0]
    disturbance_torque[0] *= 0.9

    # Save for torque plot
    torque_history.append(u)
    time_history.append(time[0])

    # Trim plot history
    while time_history and time[0] - time_history[0] > 10:
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
    ax_torque.set_xlim(max(0, time[0] - 10), time[0])
    torque_line.set_data(time_history, torque_history)
    ax_torque.relim()
    ax_torque.autoscale_view(scalex=False, scaley=True)

    return line, torque_line

ani = FuncAnimation(fig, animate, interval=50)
plt.show()
