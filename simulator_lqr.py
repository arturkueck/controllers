import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use('TkAgg')  # Only works in a GUI-enabled environment

# === Physical parameters ===
g = 9.81
L = 1.0
m = 1.0
I = m * L**2
dt = 0.05

# === LQR initial matrices ===
Q_theta_init = 10.0
Q_omega_init = 1.0
R_init = 1.0

# === LQR Gain Computation ===
def compute_lqr_gain(Q_theta, Q_omega, R):
    A = np.array([[0, 1],
                  [-g / L, 0]])
    B = np.array([[0],
                  [1 / I]])
    Q = np.diag([Q_theta, Q_omega])
    R = np.array([[R]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K.flatten()

K = compute_lqr_gain(Q_theta_init, Q_omega_init, R_init)

# === State ===
theta0 = 0.2
omega0 = 0.0
state = [theta0, omega0]
time = [0.0]
disturbance_torque = [0.0]
is_paused = False
debug_shown = False

# === Plot ===
fig, (ax_pend, ax_trq) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [2, 1]})
plt.subplots_adjust(bottom=0.4, hspace=0.4)

ax_pend.set_xlim(-L-0.2, L+0.2)
ax_pend.set_ylim(-L-0.2, L+0.2)
ax_pend.set_aspect('equal')
pend_line, = ax_pend.plot([], [], 'o-', lw=2)
pend_title = ax_pend.set_title("LQR-Controlled Pendulum")

ax_trq.set_xlim(0, 10)
ax_trq.set_ylim(-50, 50)
ax_trq.set_xlabel("Time [s]")
ax_trq.set_ylabel("Torque [Nm]")
trq_line, = ax_trq.plot([], [], color='tab:red')

torque_history = []
time_history = []

# === Sliders ===
ax_qtheta = plt.axes([0.25, 0.30, 0.6, 0.025])
ax_qomega = plt.axes([0.25, 0.25, 0.6, 0.025])
ax_r = plt.axes([0.25, 0.20, 0.6, 0.025])
s_qtheta = Slider(ax_qtheta, 'Q_theta', 0.1, 100.0, valinit=Q_theta_init)
s_qomega = Slider(ax_qomega, 'Q_omega', 0.1, 20.0, valinit=Q_omega_init)
s_r = Slider(ax_r, 'R', 0.1, 20.0, valinit=R_init)
ax_setpoint = plt.axes([0.25, 0.15, 0.6, 0.03])
slider_setpoint = Slider(ax_setpoint, 'Setpoint (°)', -180.0, 180.0, valinit=0.0)



def update_K(val=None):
    global K
    K = compute_lqr_gain(s_qtheta.val, s_qomega.val, s_r.val)
s_qtheta.on_changed(update_K)
s_qomega.on_changed(update_K)
s_r.on_changed(update_K)

# === Buttons ===
ax_l = plt.axes([0.10, 0.05, 0.15, 0.05])
ax_p = plt.axes([0.42, 0.05, 0.15, 0.05])
ax_rgt = plt.axes([0.75, 0.05, 0.15, 0.05])
b_l = Button(ax_l, 'Push Left', color='lightgray')
b_p = Button(ax_p, 'Pause', color='lightgray')
b_rgt = Button(ax_rgt, 'Push Right', color='lightgray')

def push_left(event): disturbance_torque[0] += 20.0
def push_right(event): disturbance_torque[0] -= 20.0
def toggle_pause(event):
    global is_paused, debug_shown
    is_paused = not is_paused
    if is_paused:
        show_debug()
        ani.event_source.stop()
    else:
        debug_shown = False
        ani.event_source.start()
    b_p.label.set_text("Resume" if is_paused else "Pause")

b_l.on_clicked(push_left)
b_rgt.on_clicked(push_right)
b_p.on_clicked(toggle_pause)

# === Debug Info ===
def show_debug():
    fig_d, ax_d = plt.subplots(figsize=(6, 5))
    fig_d.canvas.manager.set_window_title("LQR Equation Breakdown")
    ax_d.axis("off")

    theta, omega = state
    theta_d = np.radians(slider_setpoint.val)
    error = theta - theta_d
    u = -K @ np.array([error, omega])

    ax_d.text(0.05, 0.9, r"$u = -Kx = -[k_1, k_2] \cdot [\theta - \theta_d, \omega]^T$", fontsize=15, ha='center')
    ax_d.text(0.05, 0.35,
        f"1. θ  = {np.degrees(theta):.2f}°\n"
        f"2. ω  = {omega:.3f} rad/s\n"
        f"3. θd = {slider_setpoint.val:.1f}°\n\n"
        f"4. x  = [θ - θd, ω] = [{np.degrees(error):.2f}, {omega:.3f}]\n"
        f"5. K  = [{K[0]:.2f}, {K[1]:.2f}]\n"
        f"6. u  = -K·x = {u:.3f} Nm",
        fontsize=11, family="monospace"
    )
    plt.tight_layout()
    plt.show()

# === Animation ===
def animate(_):
    global state, time, debug_shown
    theta, omega = state
    theta_d = np.radians(slider_setpoint.val)
    error = theta - theta_d
    x = np.array([error, omega])
    torque_gain = 15.0
    u = torque_gain * (-K @ x) + disturbance_torque[0]
    disturbance_torque[0] *= 0.9

    if not is_paused:
        def ode(_, y):
            th, om = y
            return [om, -(g/L)*np.sin(th) + u/I]
        sol = solve_ivp(ode, [0, dt], state, t_eval=[dt])
        state[:] = sol.y[:, -1]
        time[0] += dt
    elif not debug_shown:
        show_debug()
        debug_shown = True

    x_tip, y_tip = L*np.sin(state[0]), -L*np.cos(state[0])
    pend_line.set_data([0, x_tip], [0, y_tip])
    pend_title.set_text(f"t={time[0]:.1f}s | θ={np.degrees(state[0]):+.1f}° | τ={u:.2f}")

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
