"""
共轴八旋翼闭环仿真演示
运行：python main.py
"""
import numpy as np
from src.simulator import Simulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    dt = 0.01
    sim = Simulator(dt=dt, mass=120.0)
    sim.reset(pos=np.array([0.0, 0.0, 10.0]))

    # 复合指令：先悬停，再前飞，再转向
    def cmd_func(t):
        if t < 5.0:
            return {'pos_neu': np.array([0.0, 0.0]), 'yaw': 0.0, 'alt': 10.0}
        elif t < 15.0:
            return {'pos_neu': np.array([20.0, 0.0]), 'yaw': 0.0, 'alt': 10.0}
        else:
            return {'pos_neu': np.array([20.0, 0.0]), 'yaw': 45.0, 'alt': 10.0}

    print("开始仿真，总时长 25s，步长 10ms ...")
    logs = sim.run(cmd_func, duration=25.0, log_interval=0.01)
    print("仿真完成，绘制结果 ...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # 高度
    ax = axes[0, 0]
    ax.plot(logs['time'], logs['pos_neu'][:, 2], label='altitude')
    ax.axhline(10.0, color='r', linestyle='--', label='cmd')
    ax.set_ylabel('Altitude (m)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True)

    # 水平位置
    ax = axes[0, 1]
    ax.plot(logs['time'], logs['pos_neu'][:, 0], label='x')
    ax.plot(logs['time'], logs['pos_neu'][:, 1], label='y')
    ax.axhline(20.0, color='r', linestyle='--')
    ax.set_ylabel('Horizontal Position (m)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True)

    # 姿态
    ax = axes[1, 0]
    ax.plot(logs['time'], logs['roll_deg'], label='roll')
    ax.plot(logs['time'], logs['pitch_deg'], label='pitch')
    ax.plot(logs['time'], logs['yaw_deg'], label='yaw')
    ax.set_ylabel('Attitude (deg)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True)

    # 速度
    ax = axes[1, 1]
    ax.plot(logs['time'], logs['vel_neu'][:, 0], label='vx')
    ax.plot(logs['time'], logs['vel_neu'][:, 1], label='vy')
    ax.plot(logs['time'], logs['vel_neu'][:, 2], label='vz')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True)

    # PWM
    ax = axes[2, 0]
    for i in range(8):
        ax.plot(logs['time'], logs['pwm'][:, i], label=f'M{i+1}')
    ax.set_ylabel('PWM')
    ax.set_xlabel('Time (s)')
    ax.legend(ncol=4, fontsize=7)
    ax.grid(True)

    # 轨迹 (俯视图)
    ax = axes[2, 1]
    ax.plot(logs['pos_neu'][:, 1], logs['pos_neu'][:, 0], 'b-', label='trajectory')
    ax.scatter([0, 20], [0, 0], color='red', marker='x', s=100, label='waypoints')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('simulation_result.png', dpi=150)
    print("结果已保存到 simulation_result.png")


if __name__ == '__main__':
    main()
