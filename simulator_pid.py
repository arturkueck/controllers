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

is_paused = False
debug_shown = False
theta_history = []

def show_pid_debug(theta, omega, dt, controller, torque):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.canvas.manager.set_window_title("PID Equation Breakdown")
    ax.axis("off")

    error = controller.setpoint - theta
    derivative = controller.last_derivative
    integral = controller.integral

    latex = (
        r"$\tau = K_p \cdot e(t) + K_i \cdot \int e(t)\,dt + K_d \cdot \frac{de(t)}{dt}$"
    )

    values = (
        f"e(t) = setpoint - θ = {np.degrees(controller.setpoint):.1f}° - {np.degrees(theta):.1f}° = {np.degrees(error):.1f}°"
        f"∫e(t)dt = {integral:.4f}\n"
        f"de(t)/dt = {derivative:.7f}\n\n"
        f"Kp = {controller.Kp:.2f}, Ki = {controller.Ki:.2f}, Kd = {controller.Kd:.2f}\n\n"
        f"τ = {torque:.4f} Nm"
    )

    ax.text(0.5, 0.9, latex, ha="center", va="top", fontsize=16)
    ax.text(0.05, 0.75, values, ha="left", va="top", fontsize=12, family="monospace")

    plt.tight_layout()
    plt.show()


# === PID Controller ===
class PIDController:
    def __init__(self, setpoint=0.0):
        self.setpoint = setpoint
        self.last_derivative = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.Kp = 10.0
        self.Ki = 0.0
        self.Kd = 2.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def get_torque(self, theta, omega, dt):
        error = self.setpoint - theta
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.last_derivative = derivative  # 🆕 Save it here
        self.integral += error * dt
        torque = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return torque


controller = PIDController()

# === State and disturbance ===
state = [theta0, omega0]
time = [0.0]
disturbance_torque = [0.0]
time_history = []
torque_history = []

# === Plot setup ===
fig, (ax_pendulum, ax_force) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.4})
plt.subplots_adjust(left=0.15, bottom=0.5)

# -- Pendulum subplot
ax_pendulum.set_xlim(-L - 0.2, L + 0.2)
ax_pendulum.set_ylim(-L - 0.2, L + 0.2)
ax_pendulum.set_aspect('equal')
pendulum_line, = ax_pendulum.plot([], [], 'o-', lw=2)
pendulum_title = ax_pendulum.set_title("Live PID-Controlled Pendulum")

# -- Torque plot
ax_force.set_xlim(0, 10)
ax_force.set_ylim(-50, 50)
ax_force.set_title("Torque over Time")
ax_force.set_xlabel("Time [s]")
ax_force.set_ylabel("Torque [Nm]")
torque_line, = ax_force.plot([], [], lw=2, color='tab:red', label="Torque")

# === Sliders ===
slider_ax_kp = plt.axes([0.25, 0.30, 0.6, 0.025])
slider_ax_ki = plt.axes([0.25, 0.25, 0.6, 0.025])
slider_ax_kd = plt.axes([0.25, 0.20, 0.6, 0.025])

slider_kp = Slider(slider_ax_kp, 'Kp', 0.0, 100.0, valinit=controller.Kp)
slider_ki = Slider(slider_ax_ki, 'Ki', 0.0, 1.0, valinit=controller.Ki)
slider_kd = Slider(slider_ax_kd, 'Kd', 0.0, 500.0, valinit=controller.Kd)

def update_gains(val=None):
    controller.Kp = slider_kp.val
    controller.Ki = slider_ki.val
    controller.Kd = slider_kd.val
    controller.setpoint = np.radians(setpoint_slider.val)  # convert degrees → radians


slider_kp.on_changed(update_gains)
slider_ki.on_changed(update_gains)
slider_kd.on_changed(update_gains)

# === Push Buttons ===
push_left_ax = plt.axes([0.1, 0.02, 0.15, 0.05])
push_right_ax = plt.axes([0.75, 0.02, 0.15, 0.05])
button_left = Button(push_left_ax, 'Push Left', color='lightgray', hovercolor='gray')
button_right = Button(push_right_ax, 'Push Right', color='lightgray', hovercolor='gray')

pause_ax      = plt.axes([0.42, 0.02, 0.15, 0.05])
pause_button = Button(pause_ax, 'Pause', color='lightgray', hovercolor='gray')

def toggle_pause(event):
    global is_paused, debug_shown
    is_paused = not is_paused
    debug_shown = False
    pause_button.label.set_text('Resume' if is_paused else 'Pause')

pause_button.on_clicked(toggle_pause)


def push_left(event):
    # push → rotate clockwise = decrease θ
    state[0] -= np.radians(angle_slider.val)

def push_right(event):
    # push → rotate counterclockwise = increase θ
    state[0] += np.radians(angle_slider.val)

button_left.on_clicked(push_left)
button_right.on_clicked(push_right)

angle_slider_ax = plt.axes([0.25, 0.10, 0.5, 0.025])
angle_slider = Slider(angle_slider_ax, 'Push Angle (°)', 0.0, 90.0, valinit=15.0)

setpoint_slider_ax = plt.axes([0.25, 0.15, 0.5, 0.025])
setpoint_slider = Slider(setpoint_slider_ax, 'Setpoint (°)', -180.0, 180.0, valinit=0.0)



# === ODE system ===
def pendulum_ode(t, y):
    theta, omega = y
    torque = controller.get_torque(theta, omega, dt) + disturbance_torque[0]
    controller.last_torque = torque  # already exists ✅
    disturbance_torque[0] *= 0.9
    dtheta_dt = omega
    domega_dt = - (g / L) * np.sin(theta) + torque / I
    return [dtheta_dt, domega_dt]

# === Animation function ===
def animate(frame):
    global state, time, debug_shown

    if is_paused:
        if not debug_shown:
            theta, omega = state
            torque = controller.last_torque
            show_pid_debug(theta, omega, dt, controller, torque)
            debug_shown = True
        return pendulum_line, torque_line


    update_gains()
    sol = solve_ivp(pendulum_ode, [0, dt], state, t_eval=[dt])
    theta, omega = sol.y[:, -1]
    state[:] = [theta, omega]
    time[0] += dt

    torque = controller.last_torque  # ✅ Reuse the computed value
    time_history.append(time[0])
    torque_history.append(torque)
    theta_history.append(theta)

    # Draw pendulum
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    pendulum_line.set_data([0, x], [0, y])
    pendulum_title.set_text(f"t={time[0]:.1f}s | θ={np.degrees(theta):.1f}° | τ={torque:.2f}")

    # Draw torque graph
    ax_force.set_xlim(max(0, time[0] - 10), time[0])
    torque_line.set_data(time_history, torque_history)
    ax_force.relim()
    ax_force.autoscale_view(scalex=False, scaley=True)

    return pendulum_line, torque_line

ani = FuncAnimation(fig, animate, interval=50)
plt.show()
