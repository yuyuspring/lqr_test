"""
前后单通道响应对比脚本。

使用同一条参考激励分别驱动 planar 与 3d 吊摆模型，
仅保留前后通道，输出对比 CSV、统计和图像。

默认复用 xp_16_7 的 lqr_comparison.csv 中 Y 轴参考波形，
并将其重映射到前后通道。这样可以保证两种模型使用完全相同的激励，
同时避免当前数据集中 X 轴参考全为 0 的问题。
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simulator import Simulator


DEFAULT_CSV_PATH = '/home/hcy/work_space/xp_16_7/src/app/planner/test/lqr_analysis/data_15/work/lqr_comparison.csv'


def select_excitation(df, source_axis):
    vel = df[f'orig_vel_{source_axis}'].to_numpy()
    acc = df[f'orig_acc_{source_axis}'].to_numpy()
    if np.max(np.abs(vel)) < 1e-9 and np.max(np.abs(acc)) < 1e-9:
        raise ValueError(f'source axis {source_axis} has no excitation in input csv')
    return vel, acc


def run_front_back_case(df, pendulum_model, source_axis='y', sim_dt=0.01, max_batches=301):
    excitation_vel, excitation_acc = select_excitation(df, source_axis)
    batches = list(df.groupby('timestamp_us'))

    sim = Simulator(
        dt=sim_dt,
        mass=120.0,
        inertia_diag=(67.0, 67.0, 75.0),
        pendulum_L=15.0,
        pendulum_M=150.0,
        trajectory_mode='va',
        pendulum_model=pendulum_model,
    )
    sim.reset(pos=np.array([0.0, 0.0, 10.0]))

    sim_steps_per_exec = max(1, int(round(0.02 / sim_dt)))
    logs = []
    sample_cursor = 0

    for batch_index, (timestamp_us, batch) in enumerate(batches, start=1):
        if batch_index > max_batches:
            break

        n = len(batch)
        vx_ref = excitation_vel[sample_cursor:sample_cursor + n]
        ax_ref = excitation_acc[sample_cursor:sample_cursor + n]
        vy_ref = np.zeros(n)
        ay_ref = np.zeros(n)
        sample_cursor += n

        state = sim.rigid_body.get_state()
        yaw = np.deg2rad(state['yaw_deg'])
        cy, sy = np.cos(yaw), np.sin(yaw)

        vx_ref_body = cy * vx_ref + sy * vy_ref
        vy_ref_body = -sy * vx_ref + cy * vy_ref
        ax_ref_body = cy * ax_ref + sy * ay_ref
        ay_ref_body = -sy * ax_ref + cy * ay_ref

        px_body = cy * state['pos_neu'][0] + sy * state['pos_neu'][1]
        py_body = -sy * state['pos_neu'][0] + cy * state['pos_neu'][1]
        vx_body = cy * state['vel_neu'][0] + sy * state['vel_neu'][1]
        vy_body = -sy * state['vel_neu'][0] + cy * state['vel_neu'][1]

        theta_x_lqr = sim.pendulum.get_lqr_theta_x()
        omega_x_lqr = sim.pendulum.get_lqr_omega_x()
        theta_y_lqr = sim.pendulum.get_lqr_theta_y()
        omega_y_lqr = sim.pendulum.get_lqr_omega_y()

        result = sim.lqr.processBatch(
            vx_ref_body, vy_ref_body, ax_ref_body, ay_ref_body,
            px0_body=px_body, vx0_body=vx_body,
            thetaBodyX0=theta_y_lqr, omegaBodyX0=omega_y_lqr,
            py0_body=py_body, vy0_body=vy_body,
            thetaBodyY0=theta_x_lqr, omegaBodyY0=omega_x_lqr,
        )

        vx_out_body = result['vxOut']
        vy_out_body = result['vyOut']
        ax_out_body = result['axOut']
        ay_out_body = result['ayOut']

        n_exec = len(vx_out_body)
        vx_out_enu = np.zeros(n_exec)
        vy_out_enu = np.zeros(n_exec)
        ax_out_enu = np.zeros(n_exec)
        ay_out_enu = np.zeros(n_exec)
        for step_idx in range(n_exec):
            vx_out_enu[step_idx] = cy * vx_out_body[step_idx] - sy * vy_out_body[step_idx]
            vy_out_enu[step_idx] = sy * vx_out_body[step_idx] + cy * vy_out_body[step_idx]
            ax_out_enu[step_idx] = cy * ax_out_body[step_idx] - sy * ay_out_body[step_idx]
            ay_out_enu[step_idx] = sy * ax_out_body[step_idx] + cy * ay_out_body[step_idx]

        for step_idx in range(n_exec):
            cmd = {
                'vel_neu': np.array([vx_out_enu[step_idx], vy_out_enu[step_idx]]),
                'acc_neu': np.array([ax_out_enu[step_idx], ay_out_enu[step_idx]]),
                'yaw': 0.0,
                'alt': 10.0,
            }
            for _ in range(sim_steps_per_exec):
                true_state, _ = sim.step(cmd)

            logs.append({
                'model': pendulum_model,
                'time_s': len(logs) * 0.02,
                'timestamp_us': timestamp_us,
                'point_idx': step_idx,
                'source_axis': source_axis,
                'orig_excitation_vel': vx_ref[step_idx] if step_idx < n else 0.0,
                'orig_excitation_acc': ax_ref[step_idx] if step_idx < n else 0.0,
                'lqr_cmd_acc_x': ax_out_body[step_idx],
                'lqr_cmd_acc_y': ay_out_body[step_idx],
                'uav_acc_body_x': true_state['accel_body'][0],
                'uav_acc_body_y': true_state['accel_body'][1],
                'uav_vel_enu_x': true_state['vel_neu'][0],
                'uav_vel_enu_y': true_state['vel_neu'][1],
                'tension_body_x': true_state['tension_n'][0],
                'tension_body_y': true_state['tension_n'][1],
                'tension_body_z': true_state['tension_n'][2],
                'gyro_angle_x': np.rad2deg(sim.pendulum.theta_x),
                'gyro_angle_y': np.rad2deg(sim.pendulum.theta_y),
                'gyro_rate_x': np.rad2deg(sim.pendulum.omega_x),
                'gyro_rate_y': np.rad2deg(sim.pendulum.omega_y),
            })

    return pd.DataFrame(logs)


def build_summary(df, windows=(15, 60)):
    rows = []
    metrics = [
        'lqr_cmd_acc_x',
        'uav_acc_body_x',
        'tension_body_x',
        'tension_body_z',
        'gyro_angle_y',
        'uav_vel_enu_x',
    ]
    for model_name, model_df in df.groupby('model'):
        for window in windows:
            view = model_df.iloc[:window]
            for metric in metrics:
                rows.append({
                    'model': model_name,
                    'window_rows': window,
                    'window_s': 0.02 * max(window - 1, 0),
                    'metric': metric,
                    'end_value': view[metric].iloc[-1],
                    'max_abs': view[metric].abs().max(),
                })
    return pd.DataFrame(rows)


def plot_comparison(df, output_path):
    planar = df[df['model'] == 'planar'].reset_index(drop=True)
    model3d = df[df['model'] == '3d'].reset_index(drop=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plots = [
        ('orig_excitation_acc', 'Excitation Acc X [m/s^2]', 'Front/Back Excitation'),
        ('uav_acc_body_x', 'Body Acc X [m/s^2]', 'UAV Body X Acceleration'),
        ('tension_body_x', 'Tension X [N]', 'Body X Tension'),
        ('tension_body_z', 'Tension Z [N]', 'Body Z Tension'),
        ('gyro_angle_y', 'Angle Y [deg]', 'Front/Back Pendulum Angle'),
        ('uav_vel_enu_x', 'ENU X Velocity [m/s]', 'ENU X Velocity'),
    ]

    for axis, (column, ylabel, title) in zip(axes.flat, plots):
        axis.plot(planar['time_s'], planar[column], label='planar', color='tab:blue')
        axis.plot(model3d['time_s'], model3d[column], label='3d', color='tab:red', alpha=0.8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel('Time [s]')
        axis.grid(True, alpha=0.3)
        axis.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def print_window_deltas(summary):
    windows = summary[['window_rows', 'window_s']].drop_duplicates().sort_values('window_rows')
    for _, window_row in windows.iterrows():
        window_rows = int(window_row['window_rows'])
        window_s = float(window_row['window_s'])
        print(f'WINDOW {window_rows} rows, {window_s:.2f}s')
        window_summary = summary[summary['window_rows'] == window_rows]
        for metric in ['uav_acc_body_x', 'tension_body_x', 'tension_body_z', 'gyro_angle_y', 'uav_vel_enu_x']:
            planar_metric = window_summary[(window_summary['model'] == 'planar') & (window_summary['metric'] == metric)].iloc[0]
            model3d_metric = window_summary[(window_summary['model'] == '3d') & (window_summary['metric'] == metric)].iloc[0]
            print(
                f"{metric}: planar_end={planar_metric['end_value']:.6f}, "
                f"3d_end={model3d_metric['end_value']:.6f}, "
                f"delta={model3d_metric['end_value'] - planar_metric['end_value']:.6f}"
            )
        print()


def main():
    parser = argparse.ArgumentParser(description='Compare planar and 3D front/back single-channel responses')
    parser.add_argument('--csv', default=DEFAULT_CSV_PATH)
    parser.add_argument('--source-axis', choices=['x', 'y'], default='y')
    parser.add_argument('--sim-dt', type=float, default=0.01)
    parser.add_argument('--max-batches', type=int, default=301)
    parser.add_argument('--output-csv', default='front_back_single_channel_comparison.csv')
    parser.add_argument('--summary-csv', default='front_back_single_channel_summary.csv')
    parser.add_argument('--output-plot', default='front_back_single_channel_comparison.png')
    args = parser.parse_args()

    df_input = pd.read_csv(args.csv)
    planar = run_front_back_case(
        df_input,
        'planar',
        source_axis=args.source_axis,
        sim_dt=args.sim_dt,
        max_batches=args.max_batches,
    )
    model3d = run_front_back_case(
        df_input,
        '3d',
        source_axis=args.source_axis,
        sim_dt=args.sim_dt,
        max_batches=args.max_batches,
    )
    comparison = pd.concat([planar, model3d], ignore_index=True)
    summary = build_summary(comparison)

    comparison.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    plot_comparison(comparison, args.output_plot)

    print('=== Front/Back Single-Channel Comparison ===')
    print(f'input csv: {args.csv}')
    print(f'source excitation axis: {args.source_axis} -> front/back channel')
    print(f'comparison csv: {args.output_csv}')
    print(f'summary csv: {args.summary_csv}')
    print(f'plot: {args.output_plot}')
    print()
    print_window_deltas(summary)


if __name__ == '__main__':
    main()
