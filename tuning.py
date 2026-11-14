import numpy as np
import main as experiment
from controllers import lqr_gain
from model import Road

road = Road()

candidates = [('pure_pursuit', float(x), np.nan, np.nan)
              for x in np.linspace(3.0, 12.0, 15)]
candidates += [('lqr', np.nan, qy, qp)
               for qy in [0.03, 0.1, 0.3, 1.0, 3.0]
               for qp in [0.3, 1.0, 3.0]]

rows = []
for index, (name, lookahead, qy, qp) in enumerate(candidates):
    K = lqr_gain(experiment.L, qy, qp, 1.0) if name == 'lqr' else None
    for speed in [12.0, 24.0]:
        _, metrics = experiment.simulate(name, speed, 0.3, road, K, lookahead)
        rows.append({
            'candidate': index,
            'controller': name,
            'lookahead_m': lookahead,
            'q_y_over_r': qy,
            'q_psi_over_r': qp,
            'speed_m_s': speed,
            **metrics
        })
    print(f'{index + 1}/{len(candidates)} {name}', flush=True)

print('\n=== Лучшие кандидаты (по RMS lateral на 12 м/с) ===\n')

for name in ['pure_pursuit', 'lqr']:
    training = [m for m in rows
                if m['controller'] == name and m['speed_m_s'] == 12.0
                and m['status'] == 'completed']

    if not training:
        print(f'{name}: нет успешных прогонов')
        continue

    best = min(training, key=lambda m: m['rms_lateral_m'])

    print(f'{name}:')
    print(f'  candidate = {best["candidate"]}')
    if name == 'pure_pursuit':
        print(f'  lookahead = {best["lookahead_m"]:.2f} m')
    else:
        print(f'  q_y/r = {best["q_y_over_r"]},  q_psi/r = {best["q_psi_over_r"]}')
    print(f'  RMS lateral (12 м/с) = {best["rms_lateral_m"]:.4f} m')
