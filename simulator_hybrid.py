import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')  # Requires tkinter


# Physical constants
g = 9.81
L = 1.0
m = 1.0
I = m * L**2
dt = 0.05

# Initial state
theta0 = 0.2
omega0 = 0.0
state = [theta0, omega0]
time = [0.0]
disturbance_torque = [0.0]
time_history, torque_history = [], []

# Controller states
a1_hat = 0.0
a2_hat = 0.0
smc_lambda = 5.0
smc_eta = 5.0
smc_eps = 0.05
is_paused = False

# Hybrid control law
def hybrid_control(theta, omega, dt):
    global a1_hat, a2_hat

    theta_d = np.radians(slider_setpoint.val)
    error = theta_d - theta
    derror = -omega

    # Feedback Linearization
    v = slider_kp.val * error + slider_kd.val * derror
    u_fl = I * (v + (g / L) * np.sin(theta))

    # Adaptive Control
    u_adapt = a1_hat * error + a2_hat * omega
    a1_hat += slider_gamma.val * error * error * dt
    a2_hat += slider_gamma.val * error * omega * dt

    # SMC
    s = omega + smc_lambda * theta
    sat_s = np.tanh(s / smc_eps)
    u_smc = -I * (g / L * np.sin(theta) + smc_lambda * omega) - smc_eta * sat_s
    smc_blend = 1.0 if abs(error) > np.radians(10) else 0.0

    return u_fl + u_adapt + smc_blend * u_smc + disturbance_torque[0]

def pendulum_ode(t, y):
    theta, omega = y
    u = hybrid_control(theta, omega, dt)
    return [omega, - (g / L) * np.sin(theta) + u / I]

# GUI
fig, (ax_pend, ax_trq) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1]})
plt.subplots_adjust(bottom=0.45, hspace=0.4)

ax_pend.set_xlim(-L-0.2, L+0.2)
ax_pend.set_ylim(-L-0.2, L+0.2)
ax_pend.set_aspect('equal')
pend_line, = ax_pend.plot([], [], 'o-', lw=2)
pend_title = ax_pend.set_title("Hybrid-Controlled Pendulum")

ax_trq.set_xlim(0, 10)
ax_trq.set_ylim(-60, 60)
ax_trq.set_xlabel("Time [s]")
ax_trq.set_ylabel("Torque [Nm]")
trq_line, = ax_trq.plot([], [], color='tab:red')

# Sliders
slider_kp = Slider(plt.axes([0.25, 0.35, 0.6, 0.025]), 'Kp', 0.0, 100.0, valinit=10.0)
slider_kd = Slider(plt.axes([0.25, 0.30, 0.6, 0.025]), 'Kd', 0.0, 50.0, valinit=2.0)
slider_gamma = Slider(plt.axes([0.25, 0.25, 0.6, 0.025]), 'γ (Adapt Speed)', 0.0, 0.5, valinit=0.1)
slider_setpoint = Slider(plt.axes([0.25, 0.20, 0.6, 0.025]), 'Setpoint (°)', -180.0, 180.0, valinit=0.0)
slider_push = Slider(plt.axes([0.25, 0.15, 0.5, 0.025]), 'Push Angle (°)', 0.0, 30.0, valinit=10.0)

# Buttons
btn_left = Button(plt.axes([0.1, 0.05, 0.15, 0.05]), 'Push Left', color='lightgray')
btn_pause = Button(plt.axes([0.42, 0.05, 0.15, 0.05]), 'Pause', color='lightgray')
btn_right = Button(plt.axes([0.75, 0.05, 0.15, 0.05]), 'Push Right', color='lightgray')

def push_left(event): state[0] -= np.radians(slider_push.val)
def push_right(event): state[0] += np.radians(slider_push.val)
def toggle_pause(event):
    global is_paused
    is_paused = not is_paused
    btn_pause.label.set_text("Resume" if is_paused else "Pause")

    if is_paused:
        show_debug()
    

btn_left.on_clicked(push_left)
btn_right.on_clicked(push_right)
btn_pause.on_clicked(toggle_pause)

def show_debug():
    theta, omega = state
    theta_d = np.radians(slider_setpoint.val)
    error = theta_d - theta
    derror = -omega

    # Recompute control contributions
    v = slider_kp.val * error + slider_kd.val * derror
    u_fl = I * (v + (g / L) * np.sin(theta))

    u_adapt = a1_hat * error + a2_hat * omega

    s = omega + smc_lambda * theta
    sat_s = np.tanh(s / smc_eps)
    u_smc = -I * (g / L * np.sin(theta) + smc_lambda * omega) - smc_eta * sat_s
    smc_blend = 1.0 if abs(error) > np.radians(10) else 0.0

    u_total = u_fl + u_adapt + smc_blend * u_smc

    # === Popup Window ===
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.canvas.manager.set_window_title("Hybrid Control Equation Breakdown")
    ax.axis("off")

    latex_main = (
        r"$u_\mathrm{total} = u_\mathrm{FL} + u_\mathrm{adaptive} + \alpha \cdot u_\mathrm{SMC}$"
    )
    latex_fl = r"$u_\mathrm{FL} = I \cdot [K_p e + K_d \dot{e} + \frac{g}{L} \sin(\theta)]$"
    latex_ad = r"$u_\mathrm{adaptive} = \hat{a}_1 \cdot e + \hat{a}_2 \cdot \omega$"
    latex_smc = r"$u_\mathrm{SMC} = -I \left( \frac{g}{L} \sin(\theta) + \lambda \cdot \omega \right) - \eta \cdot \tanh\left( \frac{s}{\varepsilon} \right)$"

    txt = (
        f"θ       = {np.degrees(theta): .2f}°\n"
        f"θd      = {np.degrees(theta_d): .2f}°\n"
        f"e       = {np.degrees(error): .2f}°\n"
        f"ω       = {omega: .3f} rad/s\n\n"

        f"u_FL       = {u_fl:.3f} Nm\n"
        f"u_adaptive = {u_adapt:.3f} Nm\n"
        f"u_SMC      = {u_smc:.3f} Nm (α = {smc_blend:.1f})\n"
        f"u_total    = {u_total:.3f} Nm\n\n"

        f"a1_hat = {a1_hat:.3f}, a2_hat = {a2_hat:.3f}\n"
        f"s = ω + λθ = {s:.3f}, tanh(s/ε) = {sat_s:.3f}"
    )

    ax.text(0.5, 0.95, latex_main, ha="center", va="top", fontsize=15)
    ax.text(0.1, 0.82, latex_fl, ha="left", va="top", fontsize=12)
    ax.text(0.1, 0.76, latex_ad, ha="left", va="top", fontsize=12)
    ax.text(0.1, 0.70, latex_smc, ha="left", va="top", fontsize=12)
    ax.text(0.1, 0.55, txt, ha="left", va="top", family="monospace", fontsize=11)

    plt.tight_layout()
    plt.show()


def animate(_):
    global state, time
    if not is_paused:
        sol = solve_ivp(pendulum_ode, [0, dt], state, t_eval=[dt])
        state[:] = sol.y[:, -1]
        time[0] += dt

    theta = state[0]
    u = hybrid_control(theta, state[1], dt)
    x_tip, y_tip = L * np.sin(theta), -L * np.cos(theta)
    pend_line.set_data([0, x_tip], [0, y_tip])
    pend_title.set_text(f"t={time[0]:.1f}s | θ={np.degrees(theta):+.1f}° | τ={u:.2f}")

    time_history.append(time[0])
    torque_history.append(u)
    while time_history and time[0] - time_history[0] > 10:
        time_history.pop(0); torque_history.pop(0)
    ax_trq.set_xlim(max(0, time[0]-10), time[0])
    trq_line.set_data(time_history, torque_history)
    ax_trq.relim(); ax_trq.autoscale_view(scalex=False, scaley=True)

    return pend_line, trq_line

ani = FuncAnimation(fig, animate, interval=50)
plt.show()
