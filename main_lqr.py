"""
LQR 消摆仿真结果可视化
对比仿真输出与原系统 CSV 数据
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_lqr_results(model='planar', sim_csv=None, output_path=None):
    sim_csv = sim_csv or (
        'lqr_comparison_sim.csv' if model == 'planar'
        else f'lqr_comparison_sim_{model}.csv')
    output_path = output_path or (
        'lqr_simulation_result.png' if model == 'planar'
        else f'lqr_simulation_result_{model}.png')

    df_sim = pd.read_csv(sim_csv)
    df_orig = pd.read_csv('/home/hcy/work_space/xp_16_7/src/app/planner/test/lqr_analysis/data_15/work/lqr_comparison.csv')

    dt = 0.02
    df_sim['time_s'] = np.arange(len(df_sim)) * dt
    df_orig['time_s'] = np.arange(len(df_orig)) * dt

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))

    ax = axes[0, 0]
    ax.plot(df_sim['time_s'], df_sim['orig_vel_x'], '--', color='tab:blue', alpha=0.7, label='vRef X')
    ax.plot(df_sim['time_s'], df_sim['orig_vel_y'], '--', color='tab:orange', alpha=0.7, label='vRef Y')
    ax.plot(df_sim['time_s'], df_sim['lqr_vel_x'], '-', color='tab:green', alpha=0.8, label='vLqr X')
    ax.plot(df_sim['time_s'], df_sim['lqr_vel_y'], '-', color='tab:red', alpha=0.8, label='vLqr Y')
    ax.plot(df_sim['time_s'], df_sim['uav_vel_enu_x'], ':', color='tab:purple', linewidth=1.5, label='vUAV X')
    ax.plot(df_sim['time_s'], df_sim['uav_vel_enu_y'], ':', color='tab:brown', linewidth=1.5, label='vUAV Y')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('ENU Velocity Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df_sim['time_s'], df_sim['lqr_acc_x'], color='tab:blue', label='lqr_acc_x (sim)')
    ax.plot(df_sim['time_s'], df_sim['lqr_acc_y'], color='tab:orange', label='lqr_acc_y (sim)')
    ax.axhline(5.0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(-5.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Accel [m/s^2]')
    ax.set_title('LQR Acceleration X/Y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df_sim['time_s'], df_sim['gyro_angle_x'], 'r-', label='gyro_angle_x (sim)')
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Pendulum Angle X (Left/Right Swing)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(df_sim['time_s'], df_sim['gyro_angle_y'], 'r-', label='gyro_angle_y (sim)')
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Pendulum Angle Y (Front/Back Swing)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(df_sim['time_s'], df_sim['lqr_integral_x'], color='tab:blue', label='integral_x (sim)')
    ax.plot(df_sim['time_s'], df_sim['lqr_integral_y'], color='tab:orange', label='integral_y (sim)')
    ax.set_ylabel('Integral')
    ax.set_title('LQR Integrator X/Y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(df_sim['time_s'], df_sim['lqr_ff_acc_x'], color='tab:blue', label='feedforward_x')
    ax.plot(df_sim['time_s'], df_sim['lqr_ff_acc_y'], color='tab:orange', label='feedforward_y')
    ax.plot(df_sim['time_s'], df_sim['lqr_swing_acc_x'], color='tab:green', linestyle='--', label='swing_uav_acc_x')
    ax.plot(df_sim['time_s'], df_sim['lqr_swing_acc_y'], color='tab:red', linestyle='--', label='swing_uav_acc_y')
    ax.set_ylabel('Accel [m/s^2]')
    ax.set_title('LQR Feedforward & Swing Acc X/Y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3, 0]
    ax.plot(df_sim['time_s'], df_sim['lqr_body_err_x'], color='tab:blue', label='err_x (sim)')
    ax.plot(df_sim['time_s'], df_sim['lqr_body_err_y'], color='tab:orange', label='err_y (sim)')
    ax.axhline(0.0, color='black', linewidth=0.5)
    ax.axhline(0.2, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(-0.2, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Error [m/s]')
    ax.set_title('LQR Body Error X/Y')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    pos_y = np.cumsum(df_sim['uav_vel_enu_y']) * dt
    pos_x = np.cumsum(df_sim['uav_vel_enu_x']) * dt
    ax.plot(pos_x, pos_y, 'b-', label='UAV trajectory (sim)')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title('UAV Trajectory (Top View)')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"结果已保存到 {output_path}")
    print()
    print("=" * 50)
    print(f"LQR 仿真统计 ({model})")
    print("=" * 50)
    print(f"总时长: {df_sim['time_s'].iloc[-1]:.2f} s")
    print(f"最大 UAV 速度 X: {df_sim['uav_vel_enu_x'].abs().max():.2f} m/s")
    print(f"最大 UAV 速度 Y: {df_sim['uav_vel_enu_y'].abs().max():.2f} m/s")
    print(f"最大摆角 X: {df_sim['gyro_angle_x'].abs().max():.2f} deg")
    print(f"最大摆角 Y: {df_sim['gyro_angle_y'].abs().max():.2f} deg")
    print(f"最大 LQR acc X: {df_sim['lqr_acc_x'].abs().max():.2f} m/s^2")
    print(f"最大 LQR acc Y: {df_sim['lqr_acc_y'].abs().max():.2f} m/s^2")
    print(f"最终 UAV 位置: ({pos_x.iloc[-1]:.2f}, {pos_y.iloc[-1]:.2f}) m")
    print(f"最终 UAV 速度: ({df_sim['uav_vel_enu_x'].iloc[-1]:.4f}, {df_sim['uav_vel_enu_y'].iloc[-1]:.4f}) m/s")
    print("=" * 50)

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Plot LQR swing simulation results')
    parser.add_argument('--model', choices=['planar', '3d'],
                        default=os.environ.get('LQR_PENDULUM_MODEL', 'planar'))
    parser.add_argument('--sim-csv', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    plot_lqr_results(args.model, args.sim_csv, args.output)


if __name__ == '__main__':
    main()
