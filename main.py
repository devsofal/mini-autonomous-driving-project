import csv
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model import Road, step
from controllers import pure_pursuit, lqr, lqr_gain
matplotlib.use('Agg')

L = 2.5
max_steer = np.radians(30)
dt = 0.01
speeds = [12.0, 18.0]
taus = [0.3, ]
initial_state = np.array([0, 0, np.radians(0), 0.0])
goal_s = 300.0
loss_lateral = 1.0
loss_heading = np.radians(60)
loss_duration = 1.0

lookahead = 3.64
q_y, q_psi, r = 0.3, 3.0, 1.0


def simulate(name, v, tau, road, K, lookahead_m=None):
    if lookahead_m is None:
        lookahead_m = lookahead

    state = initial_state.copy()
    max_steps = int(np.ceil(1.5 * goal_s / (v * dt)))
    # 13 столбцов: t, x, y, psi, delta, s, e_y, e_psi, kappa, raw, u, rate, saturated.
    data = np.full((max_steps + 1, 13), np.nan)
    violation_time = 0.0
    status = 'timeout'
    s, e_y, e_psi, curvature = road.errors(state)

    for i in range(max_steps):
        if name == 'pure_pursuit':
            raw = pure_pursuit(state, road, s, L, lookahead_m)
        else:
            raw = lqr(e_y, e_psi, curvature, L, K)
        command = np.clip(raw, -max_steer, max_steer)
        new_state = step(state, command, v, tau, dt, L)
        rate = (new_state[3] - state[3]) / dt
        saturated = abs(raw) >= max_steer - 1e-12
        data[i] = [i * dt, state[0], state[1], state[2], state[3],
                   s, e_y, e_psi, curvature, raw, command, rate, saturated]

        state = new_state
        s, e_y, e_psi, curvature = road.errors(state)
        if abs(e_y) > loss_lateral or abs(e_psi) > loss_heading:
            violation_time += dt
        else:
            violation_time = 0.0
        if violation_time >= loss_duration - 1e-12:
            status = 'lost_tracking'
            break
        if s >= goal_s:
            status = 'completed'
            break

    # Последнее состояние: следующей команды и скорости за интервал уже нет.
    n = i + 1
    data[n, :9] = [n * dt, state[0], state[1], state[2], state[3],
                   s, e_y, e_psi, curvature]
    data = data[:n + 1]
    duration = data[-1, 0]
    metrics = {
        'controller': name, 'speed_m_s': v, 'tau_s': tau, 'status': status,
        'duration_s': duration, 'final_s_m': s,
        'rms_lateral_m': np.sqrt(np.trapz(data[:, 6] ** 2, data[:, 0]) / duration),
        'max_lateral_m': np.max(np.abs(data[:, 6])),
        'max_actual_steer_deg': np.degrees(np.max(np.abs(data[:, 4]))),
        'max_command_deg': np.degrees(np.max(np.abs(data[:-1, 10]))),
        'max_rate_fd_deg_s': np.degrees(np.max(np.abs(data[:-1, 11]))),
        'rms_rate_fd_deg_s': np.degrees(np.sqrt(np.mean(data[:-1, 11] ** 2))),
        'saturation_fraction': np.mean(data[:-1, 12]),
    }
    return data, metrics


if __name__ == '__main__':
    road = Road()
    K = lqr_gain(L, q_y, q_psi, r)
    output = Path(__file__).resolve().parent / 'results'
    output.mkdir(exist_ok=True)
    runs = []
    print('LQR gain:', K)
    for name in ['pure_pursuit', 'lqr']:
        for v in speeds:
            for tau in taus:
                data, metrics = simulate(name, v, tau, road, K)
                runs.append((data, metrics))
                print(f"{name:13} v={v:g} tau={tau:g}: "
                      f"RMS={metrics['rms_lateral_m']:.4f} m, {metrics['status']}")

    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(runs[0][1]))
        writer.writeheader()
        for data, metrics in runs:
            writer.writerow(metrics)

    print("Max rate [deg/s]:", np.degrees(np.max(np.abs(data[:-1, 11]))))

    colors = ['#087e8b', '#dc6b32']
    style = {'figure.facecolor': '#f7f9fc', 'axes.facecolor': 'white',
             'text.color': '#233247', 'axes.labelcolor': '#233247',
             'xtick.color': '#526174', 'ytick.color': '#526174',
             'axes.edgecolor': '#ccd4df', 'font.size': 11}
    with plt.rc_context(style):
        for name, title in [('pure_pursuit', 'Pure Pursuit'), ('lqr', 'LQR')]:
            selected = [(d, m) for d, m in runs if m['controller'] == name]
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
            fig.suptitle(f'{title} / PATH TRACKING\n'
                         f'Actuator time constant {taus[0]:g} s  |  {goal_s:g} m experiment',
                         fontsize=21, fontweight='bold')
            trajectory, error, steer, rate = axes.flat
            mask = road.s <= max(m['final_s_m'] for d, m in selected) + 0.3
            trajectory.plot(road.x[mask], road.y[mask], color='#8795a8',
                            ls='--', lw=2, label='Reference')
            for color, (data, m) in zip(colors, selected):
                label = f"{m['speed_m_s']:g} m/s"
                trajectory.plot(data[:, 1], data[:, 2], color=color, lw=2.2, label=label)
                error.semilogy(data[:, 0], np.maximum(np.abs(data[:, 6]), 1e-8),
                               color=color, lw=2.2, label=label)
                steer.plot(data[:, 0], np.degrees(data[:, 4]), color=color,
                           lw=2.2, label=label + ' actual')
                steer.plot(data[:-1, 0], np.degrees(data[:-1, 10]), color=color,
                           lw=1.4, ls='--', alpha=0.65, label=label + ' command')
                rate.plot(data[:-1, 0] + dt / 2, np.degrees(data[:-1, 11]),
                          color=color, lw=2.2, label=label)
            trajectory.set(title='01 / Trajectories', xlabel='x [m]', ylabel='y [m]')
            trajectory.set_aspect('equal', adjustable='datalim')
            error.set(title='02 / Lateral error', xlabel='Time [s]', ylabel='|Lateral error| [m]')
            steer.set(title='03 / Steering angle', xlabel='Time [s]', ylabel='Steer [deg]')
            for limit in [-30, 30]:
                steer.axhline(limit, color='#8795a8', ls=':', lw=1)
            rate.set(title='04 / Steering rate', xlabel='Time [s]', ylabel='Steer rate [deg/s]')
            for ax in axes.flat:
                ax.grid(True, which='major', color='#e3e8ef', lw=0.8)
                ax.spines[['top', 'right']].set_visible(False)
                ax.legend(frameon=False, fontsize=9)
                panel_title = ax.get_title()
                ax.set_title('')
                ax.set_title(panel_title, loc='left', pad=12, fontweight='bold')
            fig.savefig(output / f'github_{name}.png', dpi=200)
            plt.close(fig)
    print('Results:', output)
