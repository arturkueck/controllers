import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib
matplotlib.use('TkAgg')

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
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.a1_hat = 0.0
        self.a2_hat = 0.0
        self.setpoint = 0.0

    def reset(self):
        self.a1_hat = 0.0
        self.a2_hat = 0.0

    def get_torque_and_update(self, theta, omega, dt):
        error = self.setpoint - theta
        u = self.a1_hat * theta + self.a2_hat * omega
        self.a1_hat += self.gamma * error * theta * dt
        self.a2_hat += self.gamma * error * omega * dt

        # Prevent overflow
        self.a1_hat = np.clip(self.a1_hat, -50, 50)
        self.a2_hat = np.clip(self.a2_hat, -50, 50)

        return u

controller = AdaptiveController(gamma=5.0)

# === State and logs ===
state = [theta0, omega0]
time = [0.0]
time_history = []
torque_history = []

# === Plot setup ===
fig, (ax_pendulum, ax_torque) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.4})
plt.subplots_adjust(left=0.15, bottom=0.35)  # Less vertical padding

# -- Pendulum
ax_pendulum.set_xlim(-L - 0.2, L + 0.2)
ax_pendulum.set_ylim(-L - 0.2, L + 0.2)
ax_pendulum.set_aspect('equal')
pendulum_line, = ax_pendulum.plot([], [], 'o-', lw=2)
pendulum_title = ax_pendulum.set_title("Adaptive-Controlled Pendulum")

# -- Torque plot
ax_torque.set_xlim(0, 10)
ax_torque.set_ylim(-50, 50)
ax_torque.set_title("Torque over Time")
ax_torque.set_xlabel("Time [s]")
ax_torque.set_ylabel("Torque [Nm]")
torque_line, = ax_torque.plot([], [], lw=2, color='tab:red')

# === UI Controls ===

# Gamma slider (top of controls)
gamma_slider_ax = plt.axes([0.25, 0.22, 0.5, 0.025])
gamma_slider = Slider(gamma_slider_ax, 'γ (Adapt Speed)', 0.1, 20.0, valinit=controller.gamma)

# Push buttons (middle)
push_left_ax = plt.axes([0.3, 0.15, 0.15, 0.05])
push_right_ax = plt.axes([0.55, 0.15, 0.15, 0.05])
button_left = Button(push_left_ax, 'Push Left', color='lightgray', hovercolor='gray')
button_right = Button(push_right_ax, 'Push Right', color='lightgray', hovercolor='gray')

# Push angle slider (bottom of controls)
angle_slider_ax = plt.axes([0.25, 0.07, 0.5, 0.025])
angle_slider = Slider(angle_slider_ax, 'Push Angle (°)', 0.0, 30.0, valinit=10.0)

def update_gamma(val):
    controller.gamma = gamma_slider.val

gamma_slider.on_changed(update_gamma)

def push_left(event):
    state[0] -= np.radians(angle_slider.val)

def push_right(event):
    state[0] += np.radians(angle_slider.val)

button_left.on_clicked(push_left)
button_right.on_clicked(push_right)

# === ODE system ===
def pendulum_ode(t, y):
    theta, omega = y
    torque = controller.get_torque_and_update(theta, omega, dt)
    dtheta_dt = omega
    domega_dt = - (g / L) * np.sin(theta) + torque / I
    return [dtheta_dt, domega_dt]

# === Animation ===
def animate(frame):
    global state, time

    sol = solve_ivp(pendulum_ode, [0, dt], state, t_eval=[dt])
    if not sol.success:
        return pendulum_line, torque_line

    theta, omega = sol.y[:, -1]
    state[:] = [theta, omega]
    time[0] += dt

    torque = controller.get_torque_and_update(theta, omega, dt)
    time_history.append(time[0])
    torque_history.append(torque)

    # Draw pendulum
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    pendulum_line.set_data([0, x], [0, y])
    pendulum_title.set_text(f"t={time[0]:.1f}s | θ={np.degrees(theta):.1f}° | τ={torque:.2f} | γ={controller.gamma:.1f}")

    # Update torque plot
    ax_torque.set_xlim(max(0, time[0] - 10), time[0])
    torque_line.set_data(time_history, torque_history)
    ax_torque.relim()
    ax_torque.autoscale_view(scalex=False, scaley=True)

    return pendulum_line, torque_line

ani = FuncAnimation(fig, animate, interval=50)
plt.show()
