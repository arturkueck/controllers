import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')

# ========= Physical parameters =========
g = 9.81
L = 1.0
m = 1.0
I = m * L**2
dt = 0.05

# ========= initial state / logging =========
theta0 = 0.2          # rad  (≈ 11.5°)
omega0 = 0.0
state  = [theta0, omega0]
time   = [0.0]

disturbance_torque = [0.0]
time_history, torque_history = [], []

# ========= UI flags =========
is_paused   = False
debug_shown = False

# ========= matplotlib figure =========
fig, (ax_pend, ax_trq) = plt.subplots(
        2, 1, figsize=(6, 7),
        gridspec_kw={'height_ratios': [2, 1]}
)
plt.subplots_adjust(left=0.15, bottom=0.52, hspace=0.4)

# --- Pendulum panel
ax_pend.set_xlim(-L-0.2,  L+0.2)
ax_pend.set_ylim(-L-0.2,  L+0.2)
ax_pend.set_aspect('equal')
pend_line, = ax_pend.plot([], [], 'o-', lw=2)
pend_title  = ax_pend.set_title("Feedback Linearisation-controlled Pendulum")

# --- Torque panel
ax_trq.set_xlim(0, 10)
ax_trq.set_ylim(-50, 50)
ax_trq.set_xlabel("Time [s]")
ax_trq.set_ylabel("Torque [Nm]")
trq_line, = ax_trq.plot([], [], lw=2, color='tab:red')

# ========= sliders =========
# virtual-controller gains
slider_panel = plt.axes([0.05, 0.01, 0.9, 0.35])
slider_panel.axis("off")  # Hide this panel's frame

# Define positions for sliders (in relative units inside slider_panel)
slider_height = 0.025
slider_spacing = 0.04
slider_start = 0.35  # Start below the plot
slider_width = 0.65
slider_left = 0.25


# Reassign sliders
s_kp       = Slider(plt.axes([slider_left, slider_start, slider_width, slider_height]), "Kp", 0.0, 100.0, valinit=10.0)
s_kd       = Slider(plt.axes([slider_left, slider_start - 1*slider_spacing, slider_width, slider_height]), "Kd", 0.0, 20.0, valinit=2.0)
s_set      = Slider(plt.axes([slider_left, slider_start - 2*slider_spacing, slider_width, slider_height]), "Setpoint (°)", -180.0, 180.0, valinit=0.0)
s_m_model  = Slider(plt.axes([slider_left, slider_start - 3*slider_spacing, slider_width, slider_height]), "Model Mass m̂ [kg]", 0.5, 2.0, valinit=m)
s_L_model  = Slider(plt.axes([slider_left, slider_start - 4*slider_spacing, slider_width, slider_height]), "Model Length L̂ [m]", 0.5, 2.0, valinit=L)


# ========= push disturbance buttons =========
push_L_ax = plt.axes([0.10, 0.05, 0.15, 0.05])
pause_ax  = plt.axes([0.42, 0.05, 0.15, 0.05])
push_R_ax = plt.axes([0.75, 0.05, 0.15, 0.05])
b_left  = Button(push_L_ax,  "Push Left",  color='lightgray', hovercolor='gray')
b_pause = Button(pause_ax,   "Pause",      color='lightgray', hovercolor='gray')
b_right = Button(push_R_ax,  "Push Right", color='lightgray', hovercolor='gray')

def push_left(event):   disturbance_torque[0] += 20.0
def push_right(event):  disturbance_torque[0] -= 20.0
b_left.on_clicked(push_left)
b_right.on_clicked(push_right)

def toggle_pause(event):
    global is_paused, debug_shown
    is_paused = not is_paused

    if is_paused:
        theta, omega = state
        u, v, err, derr = fl_control(theta, omega)
        show_fl_debug(theta, omega, u, v, err, derr)
        debug_shown = True
        ani.event_source.stop()
    else:
        debug_shown = False
        ani.event_source.start()

    b_pause.label.set_text('Resume' if is_paused else 'Pause')


b_pause.on_clicked(toggle_pause)

# ========= Feedback-linearisation control =========
def fl_control(theta, omega):
    Kp, Kd = s_kp.val, s_kd.val
    theta_d = np.radians(s_set.val)
    error   = theta_d - theta
    derror  = -omega

    # Use model (possibly wrong) parameters
    m_model = s_m_model.val
    L_model = s_L_model.val
    I_model = m_model * L_model**2

    v = Kp * error + Kd * derror
    u = I_model * (v + (g / L_model) * np.sin(theta))
    return u, v, error, derror

# ========= debug popup =========
def show_fl_debug(theta, omega, u, v, err, derr):
    fig_d, ax_d = plt.subplots(figsize=(6, 5))
    fig_d.canvas.manager.set_window_title("FL Equation Breakdown")
    ax_d.axis("off")

    # Use only matplotlib-compatible mathtext
    latex = (
        r"$u = I[K_p(\theta_d - \theta) + K_d(-\omega) + \frac{g}{L}\sin(\theta)]$"
    )

    txt = (
        f"1. θ (actual angle)        = {np.degrees(theta): .2f}°\n"
        f"2. ω (angular velocity)    = {omega: .3f} rad/s\n"
        f"3. θd (desired angle)      = {s_set.val: .1f}°\n\n"

        f"4. e(t) = θd - θ           = {np.degrees(err): .2f}°\n"
        f"5. de(t)/dt = -ω          = {derr: .3f} rad/s\n\n"

        f"6. v = Kp·e + Kd·de(t)/dt\n"
        f"   = {s_kp.val:.2f}×{np.degrees(err):.2f} + {s_kd.val:.2f}×({derr:.2f})\n"
        f"   = {v:.3f}   (virtual control)\n\n"

        f"7. u = I·[v + (g/L)·sin(θ)]\n"
        f"   = {I:.2f}×[{v:.3f} + {(g/L):.2f}×sin({np.degrees(theta):.2f}°)]\n"
        f"   = {u:.3f} Nm (actual torque)"
    )

    ax_d.text(0.2, 0.9, latex, ha="center", va="top", fontsize=15)
    ax_d.text(0.1, 0.70, txt, ha="left", va="top", family="monospace", fontsize=11)
    plt.tight_layout()
    plt.show()

# ========= animation loop =========
def animate(_):
    global state, time, debug_shown

    theta, omega = state
    u, v, err, derr = fl_control(theta, omega)
    u += disturbance_torque[0]
    disturbance_torque[0] *= 0.9       # decay disturbance

    # integrate one step
    def ode(_, y):      # y = [theta, omega]
        th, om = y
        dth  = om
        dom  = -(g/L)*np.sin(th) + u/I
        return [dth, dom]

    sol      = solve_ivp(ode, [0, dt], state, t_eval=[dt])
    theta, omega = sol.y[:, -1]
    state[:] = [theta, omega]
    time[0] += dt

    # pause behaviour
    if is_paused:
        if not debug_shown:
            show_fl_debug(theta, omega, u, v, err, derr)
            debug_shown = True
        return pend_line, trq_line

    # keep 10-s history
    time_history.append(time[0])
    torque_history.append(u)
    while time_history and time[0] - time_history[0] > 10:
        time_history.pop(0); torque_history.pop(0)

    # --- draw pendulum
    x_tip, y_tip = L*np.sin(theta), -L*np.cos(theta)
    pend_line.set_data([0, x_tip], [0, y_tip])
    pend_title.set_text(f"t={time[0]:.1f}s | θ={np.degrees(theta):+.1f}° | τ={u:.2f}")

    # --- draw torque trace
    ax_trq.set_xlim(max(0, time[0]-10), time[0])
    trq_line.set_data(time_history, torque_history)
    ax_trq.relim(); ax_trq.autoscale_view(scalex=False, scaley=True)

    return pend_line, trq_line

ani = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
plt.show()
