import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use('TkAgg')

# === Physical parameters ===
g = 9.81
L = 1.0
m = 1.0
I = m * L**2
dt = 0.05

# === Initial LQR matrices ===
Q_theta_init = 10.0
Q_omega_init = 1.0
R_init = 1.0

# === LQR gain computation (downward position) ===
def compute_lqr_gain(Q_theta, Q_omega, R):
    A = np.array([[0, 1],
                  [-g / L, 0]])  # downward linearization
    B = np.array([[0],
                  [1 / I]])
    Q = np.diag([Q_theta, Q_omega])
    R = np.array([[R]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K.flatten()

K = compute_lqr_gain(Q_theta_init, Q_omega_init, R_init)

# === Initial state ===
theta0 = 0.2  # slight deviation from hanging
omega0 = 0.0
state = [theta0, omega0]
time = [0.0]
disturbance_torque = [0.0]  # external force

# === Torque history
torque_history = []
time_history = []

# === Plot setup ===
fig, (ax_pendulum, ax_torque) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1]})
plt.subplots_adjust(bottom=0.35, hspace=0.4)

# -- Pendulum drawing
ax_pendulum.set_xlim(-L - 0.2, L + 0.2)
ax_pendulum.set_ylim(-L - 0.2, L + 0.2)
ax_pendulum.set_aspect('equal')
line, = ax_pendulum.plot([], [], 'o-', lw=2)
title = ax_pendulum.set_title("LQR-Controlled Downward Pendulum with Torque Plot")

# -- Torque plot
ax_torque.set_xlim(0, 10)
ax_torque.set_ylim(-20, 20)
ax_torque.set_xlabel("Time [s]")
ax_torque.set_ylabel("Torque [Nm]")
torque_line, = ax_torque.plot([], [], lw=2, color='tab:red')

# === Sliders
ax_qtheta = plt.axes([0.25, 0.25, 0.6, 0.03])
ax_qomega = plt.axes([0.25, 0.20, 0.6, 0.03])
ax_r = plt.axes([0.25, 0.15, 0.6, 0.03])

slider_qtheta = Slider(ax_qtheta, 'Q_theta', 0.1, 100.0, valinit=Q_theta_init)
slider_qomega = Slider(ax_qomega, 'Q_omega', 0.1, 20.0, valinit=Q_omega_init)
slider_r = Slider(ax_r, 'R', 0.1, 20.0, valinit=R_init)

def update_gains(val=None):
    global K
    Q_theta = slider_qtheta.val
    Q_omega = slider_qomega.val
    R = slider_r.val
    K = compute_lqr_gain(Q_theta, Q_omega, R)

slider_qtheta.on_changed(update_gains)
slider_qomega.on_changed(update_gains)
slider_r.on_changed(update_gains)

# === Push Buttons
push_left_ax = plt.axes([0.3, 0.05, 0.15, 0.05])
push_right_ax = plt.axes([0.55, 0.05, 0.15, 0.05])

button_left = Button(push_left_ax, 'Push Left', color='lightgray', hovercolor='gray')
button_right = Button(push_right_ax, 'Push Right', color='lightgray', hovercolor='gray')

def push_left(event):
    disturbance_torque[0] += 20.0  # simulate push from right

def push_right(event):
    disturbance_torque[0] -= 20.0  # simulate push from left

button_left.on_clicked(push_left)
button_right.on_clicked(push_right)

# === Animation function ===
def animate(frame):
    global state, time

    theta, omega = state
    x = np.array([theta, omega])  # stabilization around θ = 0

    # Control input
    u = -K @ x + disturbance_torque[0]
    disturbance_torque[0] *= 0.9  # decay force

    # Store for plot
    torque_history.append(u)
    time_history.append(time[0])

    # Trim old data (keep last 10 seconds)
    max_window = 10
    while time_history and time[0] - time_history[0] > max_window:
        time_history.pop(0)
        torque_history.pop(0)

    # Integrate dynamics
    def ode(t, y):
        theta, omega = y
        dtheta_dt = omega
        domega_dt = - (g / L) * np.sin(theta) + u / I
        return [dtheta_dt, domega_dt]

    sol = solve_ivp(ode, [0, dt], state, t_eval=[dt])
    theta, omega = sol.y[:, -1]
    state[:] = [theta, omega]
    time[0] += dt

    # Draw pendulum
    x_tip = L * np.sin(theta)
    y_tip = -L * np.cos(theta)
    line.set_data([0, x_tip], [0, y_tip])
    title.set_text(f"t={time[0]:.1f}s | θ={np.degrees(theta):+.1f}° | τ={u:.2f}")

    # Update torque plot
    ax_torque.set_xlim(max(0, time[0] - max_window), time[0])
    torque_line.set_data(time_history, torque_history)
    ax_torque.relim()
    ax_torque.autoscale_view(scalex=False, scaley=True)

    return line, torque_line

ani = FuncAnimation(fig, animate, interval=50)
plt.show()
