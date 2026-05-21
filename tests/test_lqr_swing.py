"""
LQR 消摆仿真测试
使用 data_15 的 CSV 数据作为 vRef，在线运行 LQR + 6DOF + 吊摆耦合模型
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from main_lqr import plot_lqr_results
from src.simulator import Simulator


def test_lqr_swing(pendulum_model=None):
    """主测试：读取 CSV，逐 batch 执行 LQR + 6DOF 仿真"""
    if pendulum_model is None:
        pendulum_model = os.environ.get('LQR_PENDULUM_MODEL', 'planar')
    sim_dt = 0.01
    csv_path = '/home/hcy/work_space/xp_16_7/src/app/planner/test/lqr_analysis/data_15/work/lqr_comparison.csv'
    df = pd.read_csv(csv_path)
    df_swapped = df.copy()
    df_swapped[['orig_vel_x', 'orig_vel_y']] = df_swapped[['orig_vel_y', 'orig_vel_x']].to_numpy()
    df_swapped[['orig_acc_x', 'orig_acc_y']] = df_swapped[['orig_acc_y', 'orig_acc_x']].to_numpy()
    timestamp_step = int(df['timestamp_us'].diff().dropna().mode().iloc[0])
    df_swapped['timestamp_us'] = df_swapped['timestamp_us'] + int(df['timestamp_us'].max()) + timestamp_step
    df = pd.concat([df, df_swapped], ignore_index=True)

    # 按 timestamp_us 分组
    batches = df.groupby('timestamp_us')
    total_batches = len(batches)
    print(f"Total batches: {total_batches}")

    # 初始化仿真器
    sim = Simulator(
        dt=sim_dt,
        mass=120.0,
        inertia_diag=(67.0, 67.0, 75.0),
        pendulum_L=15.0,
        pendulum_M=150.0,
        trajectory_mode='va',
        pendulum_model=pendulum_model,
    )
    sim_steps_per_exec = max(1, int(round(0.02 / sim_dt)))
    sim.reset(pos=np.array([0.0, 0.0, 10.0]))

    # 日志列表
    logs = []
    batch_count = 0

    for ts, batch in batches:
        batch_count += 1
        if batch_count > total_batches:
            break

        # 读取 vRef / aRef（ENU）
        n = len(batch)
        vx_ref = batch['orig_vel_x'].values
        vy_ref = batch['orig_vel_y'].values
        ax_ref = batch['orig_acc_x'].values
        ay_ref = batch['orig_acc_y'].values

        # 获取当前 UAV 状态
        state = sim.rigid_body.get_state()
        yaw = np.deg2rad(state['yaw_deg'])
        cy, sy = np.cos(yaw), np.sin(yaw)

        # ENU → Body
        vx_ref_body = cy * vx_ref + sy * vy_ref
        vy_ref_body = -sy * vx_ref + cy * vy_ref
        ax_ref_body = cy * ax_ref + sy * ay_ref
        ay_ref_body = -sy * ax_ref + cy * ay_ref

        # 当前 UAV 位置和速度 → Body
        px_body = cy * state['pos_neu'][0] + sy * state['pos_neu'][1]
        py_body = -sy * state['pos_neu'][0] + cy * state['pos_neu'][1]
        vx_body = cy * state['vel_neu'][0] + sy * state['vel_neu'][1]
        vy_body = -sy * state['vel_neu'][0] + cy * state['vel_neu'][1]

        # 当前摆状态 → LQR 格式
        theta_x_lqr = sim.pendulum.get_lqr_theta_x()
        omega_x_lqr = sim.pendulum.get_lqr_omega_x()
        theta_y_lqr = sim.pendulum.get_lqr_theta_y()
        omega_y_lqr = sim.pendulum.get_lqr_omega_y()

        # LQR 批量处理
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
        debug = result['debug']

        # Body → ENU
        n_exec = len(vx_out_body)
        vx_out_enu = np.zeros(n_exec)
        vy_out_enu = np.zeros(n_exec)
        ax_out_enu = np.zeros(n_exec)
        ay_out_enu = np.zeros(n_exec)
        for i in range(n_exec):
            vx_out_enu[i] = cy * vx_out_body[i] - sy * vy_out_body[i]
            vy_out_enu[i] = sy * vx_out_body[i] + cy * vy_out_body[i]
            ax_out_enu[i] = cy * ax_out_body[i] - sy * ay_out_body[i]
            ay_out_enu[i] = sy * ax_out_body[i] + cy * ay_out_body[i]

        # 每个 LQR 输出点对应 20ms 仿真时间，子步数按 sim_dt 自动换算。
        for step_idx in range(n_exec):
            cmd = {
                'vel_neu': np.array([vx_out_enu[step_idx], vy_out_enu[step_idx]]),
                'acc_neu': np.array([ax_out_enu[step_idx], ay_out_enu[step_idx]]),
                'yaw': 0.0,
                'alt': 10.0,
            }
            for _ in range(sim_steps_per_exec):
                true_state, pwm = sim.step(cmd)

            # 记录日志（与原系统字段同号）
            d = debug[step_idx] if step_idx < len(debug) else {'x': None, 'y': None}
            mx = d['x'] if d['x'] is not None else None
            my = d['y'] if d['y'] is not None else None

            log_row = {
                'timestamp_us': ts,
                'point_idx': step_idx,
                'orig_vel_x': vx_ref[step_idx] if step_idx < n else 0.0,
                'orig_vel_y': vy_ref[step_idx] if step_idx < n else 0.0,
                'orig_acc_x': ax_ref[step_idx] if step_idx < n else 0.0,
                'orig_acc_y': ay_ref[step_idx] if step_idx < n else 0.0,
                'orig_body_vel_x': vx_ref_body[step_idx] if step_idx < n else 0.0,
                'orig_body_vel_y': vy_ref_body[step_idx] if step_idx < n else 0.0,
                'orig_body_acc_x': ax_ref_body[step_idx] if step_idx < n else 0.0,
                'orig_body_acc_y': ay_ref_body[step_idx] if step_idx < n else 0.0,
                'traj_vel_hori_limit': 13.5,
                'flight_lim_vel': 13.5,
                'manual_speed_limit': 13.5,
                'uav_vel_enu_x': true_state['vel_neu'][0],
                'uav_vel_enu_y': true_state['vel_neu'][1],
                'uav_vel_body_x': cy * true_state['vel_neu'][0] + sy * true_state['vel_neu'][1],
                'uav_vel_body_y': -sy * true_state['vel_neu'][0] + cy * true_state['vel_neu'][1],
                'uav_acc_body_x': true_state['accel_body'][0],
                'uav_acc_body_y': true_state['accel_body'][1],
                'uav_acc_body_z': true_state['accel_body'][2],
                'tension_body_x': true_state['tension_n'][0],
                'tension_body_y': true_state['tension_n'][1],
                'tension_body_z': true_state['tension_n'][2],
                'stick_roll': 0,
                'stick_pitch': 0,
                'stick_yaw': 0,
                'stick_throttle': 0,
                'lqr_body_vel_x': vx_out_body[step_idx] if step_idx < n_exec else 0.0,
                'lqr_body_vel_y': vy_out_body[step_idx] if step_idx < n_exec else 0.0,
                'lqr_body_vref_x': d.get('vxRef', 0.0),
                'lqr_body_vref_y': d.get('vyRef', 0.0),
                'lqr_body_err_x': (vx_out_body[step_idx] - d.get('vxRef', 0.0)) if step_idx < n_exec else 0.0,
                'lqr_body_err_y': (vy_out_body[step_idx] - d.get('vyRef', 0.0)) if step_idx < n_exec else 0.0,
                'lqr_vel_x': vx_out_enu[step_idx] if step_idx < n_exec else 0.0,
                'lqr_vel_y': vy_out_enu[step_idx] if step_idx < n_exec else 0.0,
                'lqr_acc_x': ax_out_enu[step_idx] if step_idx < n_exec else 0.0,
                'lqr_acc_y': ay_out_enu[step_idx] if step_idx < n_exec else 0.0,
                'lqr_body_target_acc_x': mx.targetAcc if mx else 0.0,
                'lqr_body_target_acc_y': my.targetAcc if my else 0.0,
                'lqr_body_cmd_acc_x': ax_out_body[step_idx] if step_idx < n_exec else 0.0,
                'lqr_body_cmd_acc_y': ay_out_body[step_idx] if step_idx < n_exec else 0.0,
                'lqr_ref_acc_x': mx.axRef if mx else 0.0,
                'lqr_ref_acc_y': my.axRef if my else 0.0,
                'lqr_ff_acc_x': mx.feedforwardAcc if mx else 0.0,
                'lqr_ff_acc_y': my.feedforwardAcc if my else 0.0,
                'lqr_swing_acc_x': mx.swingUavAcc if mx else 0.0,
                'lqr_swing_acc_y': my.swingUavAcc if my else 0.0,
                'lqr_int_before_x': mx.integralBefore if mx else 0.0,
                'lqr_int_before_y': my.integralBefore if my else 0.0,
                'lqr_int_preview_x': mx.integralPreview if mx else 0.0,
                'lqr_int_preview_y': my.integralPreview if my else 0.0,
                'lqr_integral_x': mx.integralAfter if mx else 0.0,
                'lqr_integral_y': my.integralAfter if my else 0.0,
                'lqr_aw_gain_x': mx.antiWindupGain if mx else 0.0,
                'lqr_aw_gain_y': my.antiWindupGain if my else 0.0,
                'lqr_aw_delta_x': mx.antiWindupDelta if mx else 0.0,
                'lqr_aw_delta_y': my.antiWindupDelta if my else 0.0,
                'lqr_aw_correction_x': mx.antiWindupCorrection if mx else 0.0,
                'lqr_aw_correction_y': my.antiWindupCorrection if my else 0.0,
                'lqr_aw_u_preview_x': mx.antiWindupPreviewAcc if mx else 0.0,
                'lqr_aw_u_preview_y': my.antiWindupPreviewAcc if my else 0.0,
                'lqr_aw_u_sat_x': mx.antiWindupPreviewSatAcc if mx else 0.0,
                'lqr_aw_u_sat_y': my.antiWindupPreviewSatAcc if my else 0.0,
                'gyro_angle_x': np.rad2deg(sim.pendulum.theta_x),
                'gyro_angle_y': np.rad2deg(sim.pendulum.theta_y),
                'gyro_rate_x': np.rad2deg(sim.pendulum.omega_x),
                'gyro_rate_y': np.rad2deg(sim.pendulum.omega_y),
                'rope': 15.0,
                'payload_mass': 150.0,
                'drone_mass': 120.0,
                'yaw': state['yaw_deg'],
            }
            logs.append(log_row)

        if batch_count % 10 == 0:
            print(f"  Processed batch {batch_count}/{total_batches}")

    # 输出 CSV
    df_log = pd.DataFrame(logs)
    output_path = 'lqr_comparison_sim.csv' if pendulum_model == 'planar' else f'lqr_comparison_sim_{pendulum_model}.csv'
    df_log.to_csv(output_path, index=False)
    print(f"\n[PASS] test_lqr_swing: {len(logs)} rows written to {output_path}")
    plot_path = plot_lqr_results(pendulum_model, sim_csv=output_path)
    print(f"[PASS] plot generated: {plot_path}")

    # 基本检查
    assert len(logs) > 0, "No logs generated"
    assert df_log['lqr_acc_x'].abs().max() < 10.0, "LQR x accel too large"
    assert df_log['lqr_acc_y'].abs().max() < 10.0, "LQR y accel too large"

    return df_log


if __name__ == '__main__':
    print("=" * 50)
    print("Running LQR swing test...")
    print("=" * 50)
    test_lqr_swing()
    print("=" * 50)
    print("Test PASSED")
    print("=" * 50)
