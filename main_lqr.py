"""
LQR 消摆仿真结果可视化
对比仿真输出与原系统 CSV 数据
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    df_sim = pd.read_csv('lqr_comparison_sim.csv')
    df_orig = pd.read_csv('/home/hcy/work_space/xp_16_7/src/app/planner/test/lqr_analysis/data_15/work/lqr_comparison.csv')

    dt = 0.02
    df_sim['time_s'] = np.arange(len(df_sim)) * dt
    df_orig['time_s'] = np.arange(len(df_orig)) * dt

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))

    # 1. 水平速度（ENU Y，因为数据主要是 Y 方向运动）
    ax = axes[0, 0]
    ax.plot(df_sim['time_s'], df_sim['orig_vel_y'], 'b--', alpha=0.7, label='vRef (orig)')
    ax.plot(df_sim['time_s'], df_sim['lqr_vel_y'], 'g-', alpha=0.7, label='vLqr (sim)')
    ax.plot(df_sim['time_s'], df_sim['uav_vel_enu_y'], 'r-', label='vUAV (sim)')
    ax.set_ylabel('Velocity Y [m/s]')
    ax.set_title('ENU Y Velocity Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. LQR 加速度（ENU Y）
    ax = axes[0, 1]
    ax.plot(df_sim['time_s'], df_sim['lqr_acc_y'], 'g-', label='lqr_acc_y (sim)')
    ax.plot(df_orig['time_s'], df_orig['lqr_acc_y'], 'b--', alpha=0.5, label='lqr_acc_y (orig)')
    ax.axhline(5.0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(-5.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Accel Y [m/s^2]')
    ax.set_title('LQR Acceleration Y (Sim vs Orig)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 摆角 X（左右摆）
    ax = axes[1, 0]
    ax.plot(df_sim['time_s'], df_sim['gyro_angle_x'], 'r-', label='gyro_angle_x (sim)')
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Pendulum Angle X (Left/Right Swing)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 摆角 Y（前后摆）
    ax = axes[1, 1]
    ax.plot(df_sim['time_s'], df_sim['gyro_angle_y'], 'r-', label='gyro_angle_y (sim)')
    ax.set_ylabel('Angle [deg]')
    ax.set_title('Pendulum Angle Y (Front/Back Swing)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 积分器状态
    ax = axes[2, 0]
    ax.plot(df_sim['time_s'], df_sim['lqr_integral_y'], 'purple', label='integral_y (sim)')
    ax.set_ylabel('Integral')
    ax.set_title('LQR Integrator Y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 前馈 vs 反馈
    ax = axes[2, 1]
    ax.plot(df_sim['time_s'], df_sim['lqr_ff_acc_y'], 'g-', label='feedforward_y')
    ax.plot(df_sim['time_s'], df_sim['lqr_swing_acc_y'], 'orange', label='swing_uav_acc_y')
    ax.set_ylabel('Accel [m/s^2]')
    ax.set_title('LQR Feedforward & Swing Acc Y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. 误差
    ax = axes[3, 0]
    ax.plot(df_sim['time_s'], df_sim['lqr_body_err_y'], 'b-', label='err_y (sim)')
    ax.axhline(0.0, color='black', linewidth=0.5)
    ax.axhline(0.2, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(-0.2, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Error [m/s]')
    ax.set_title('LQR Body Error Y')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. UAV 轨迹（俯视图）
    ax = axes[3, 1]
    # 积分速度得到近似位置
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
    plt.savefig('lqr_simulation_result.png', dpi=150)
    print("结果已保存到 lqr_simulation_result.png")

    # 打印统计
    print()
    print("=" * 50)
    print("LQR 仿真统计")
    print("=" * 50)
    print(f"总时长: {df_sim['time_s'].iloc[-1]:.2f} s")
    print(f"最大 UAV 速度 Y: {df_sim['uav_vel_enu_y'].abs().max():.2f} m/s")
    print(f"最大摆角 X: {df_sim['gyro_angle_x'].abs().max():.2f} deg")
    print(f"最大摆角 Y: {df_sim['gyro_angle_y'].abs().max():.2f} deg")
    print(f"最大 LQR acc Y: {df_sim['lqr_acc_y'].abs().max():.2f} m/s^2")
    print(f"最终 UAV 位置: ({pos_x.iloc[-1]:.2f}, {pos_y.iloc[-1]:.2f}) m")
    print(f"最终 UAV 速度: ({df_sim['uav_vel_enu_x'].iloc[-1]:.4f}, {df_sim['uav_vel_enu_y'].iloc[-1]:.4f}) m/s")
    print("=" * 50)


if __name__ == '__main__':
    main()
