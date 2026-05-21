"""
系统级单元测试
验证：悬停稳定性、位置阶跃、高度阶跃、偏航阶跃
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.simulator import Simulator


def test_hover():
    """零指令悬停：初始在 10m 高度，期望稳定悬停"""
    sim = Simulator(dt=0.001, mass=120.0)
    sim.reset(pos=np.array([0.0, 0.0, 10.0]))

    def cmd_func(t):
        return {'pos_ned': np.array([0.0, 0.0]), 'yaw': 0.0, 'alt': 10.0}

    logs = sim.run(cmd_func, duration=15.0, log_interval=0.01)

    # 检查最终高度
    final_alt = logs['pos_ned'][-1, 2]
    assert abs(final_alt - 10.0) < 0.5, f"悬停高度偏差过大: {final_alt:.2f}m"

    # 检查最终速度
    final_vel = logs['vel_ned'][-1]
    assert np.linalg.norm(final_vel) < 0.5, f"悬停速度过大: {final_vel}"

    print("[PASS] test_hover")
    return logs


def test_step_altitude():
    """高度阶跃：0~5s 指令 10m，5~15s 指令 20m"""
    sim = Simulator(dt=0.001, mass=120.0)
    sim.reset(pos=np.array([0.0, 0.0, 10.0]))

    def cmd_func(t):
        alt = 20.0 if t >= 5.0 else 10.0
        return {'pos_ned': np.array([0.0, 0.0]), 'yaw': 0.0, 'alt': alt}

    logs = sim.run(cmd_func, duration=15.0, log_interval=0.01)

    # 检查第二次阶跃后的稳态
    final_alt = logs['pos_ned'][-1, 2]
    assert abs(final_alt - 20.0) < 1.0, f"高度阶跃稳态偏差: {final_alt:.2f}m"
    print("[PASS] test_step_altitude")
    return logs


def test_step_position():
    """水平位置阶跃：0~5s 指令 (0,0)，5~15s 指令 (10,0)"""
    sim = Simulator(dt=0.001, mass=120.0)
    sim.reset(pos=np.array([0.0, 0.0, 10.0]))

    def cmd_func(t):
        pos = np.array([10.0, 0.0]) if t >= 5.0 else np.array([0.0, 0.0])
        return {'pos_ned': pos, 'yaw': 0.0, 'alt': 10.0}

    logs = sim.run(cmd_func, duration=20.0, log_interval=0.01)

    final_pos = logs['pos_ned'][-1, :2]
    assert np.linalg.norm(final_pos - np.array([10.0, 0.0])) < 1.0, \
        f"位置阶跃稳态偏差: {final_pos}"
    print("[PASS] test_step_position")
    return logs


def test_step_yaw():
    """偏航阶跃：0~5s 指令 0deg，5~15s 指令 45deg"""
    sim = Simulator(dt=0.001, mass=120.0)
    sim.reset(pos=np.array([0.0, 0.0, 10.0]))

    def cmd_func(t):
        yaw = 45.0 if t >= 5.0 else 0.0
        return {'pos_ned': np.array([0.0, 0.0]), 'yaw': yaw, 'alt': 10.0}

    logs = sim.run(cmd_func, duration=15.0, log_interval=0.01)

    final_yaw = logs['yaw_deg'][-1]
    # 归一化到 [-180,180]
    yaw_err = ((final_yaw - 45.0 + 180.0) % 360.0) - 180.0
    assert abs(yaw_err) < 5.0, f"偏航阶跃稳态偏差: {yaw_err:.2f}deg"
    print("[PASS] test_step_yaw")
    return logs


if __name__ == '__main__':
    print("=" * 50)
    print("Running system tests...")
    print("=" * 50)
    test_hover()
    test_step_altitude()
    test_step_position()
    test_step_yaw()
    print("=" * 50)
    print("All tests PASSED")
    print("=" * 50)
