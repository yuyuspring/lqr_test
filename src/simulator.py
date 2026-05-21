"""
仿真器主循环
集成控制器、执行器、刚体动力学、吊摆、LQR 消摆规划
"""
import numpy as np
from src.controller import Controller
from src.model import ActuatorModel, RigidBodyModel
from src.pendulum import PendulumModel
from src.lqr.lqr_swing_controller import LqrSwingController


class Simulator:
    def __init__(self, dt=0.001, mass=120.0, inertia_diag=(67.0, 67.0, 75.0),
                 pendulum_L=15.0, pendulum_M=150.0,
                 trajectory_mode='pva'):
        self.dt = dt
        self.mass = mass
        self.actuator = ActuatorModel(dt)
        hover_pwm = self.actuator.compute_hover_pwm(mass * 9.81)
        self.controller = Controller(dt, hover_throttle=hover_pwm,
                                     trajectory_mode=trajectory_mode)
        self.rigid_body = RigidBodyModel(mass, inertia_diag, dt)
        self.pendulum = PendulumModel(L=pendulum_L, M_payload=pendulum_M,
                                       M_drone=mass)
        self.lqr = LqrSwingController(dt=0.02, ropeLength=pendulum_L,
                                       payloadMass=pendulum_M, droneMass=mass)
        self.time = 0.0
        self.trajectory_mode = trajectory_mode

    def reset(self, pos=None, vel=None, q=None, omega=None,
              theta_x=0.0, omega_x=0.0, theta_y=0.0, omega_y=0.0):
        self.controller.reset()
        self.actuator.reset()
        # 初始化电机转速为悬停稳态转速
        hover_pwm = self.actuator.compute_hover_pwm(self.mass * 9.81)
        omega_hover = hover_pwm / 1000.0 * self.actuator.max_omega
        self.actuator.omega = np.full(8, omega_hover)
        # ENU 坐标系下初始位置
        if pos is not None:
            pos = pos.copy()
        else:
            pos = np.zeros(3)
        self.rigid_body.reset(pos, vel, q, omega)
        self.pendulum.reset(theta_x, omega_x, theta_y, omega_y)
        self.time = 0.0

    def step(self, cmd):
        """
        cmd: dict
            pva 模式: {'pos_ned': ..., 'yaw': ..., 'alt': ...}
            va  模式: {'vel_ned': ..., 'acc_ned': ..., 'yaw': ..., 'alt': ...}
        """
        # 1. 获取真实状态作为反馈
        true_state = self.rigid_body.get_state()
        true_state['q'] = self.rigid_body.q.copy()

        # 2. 控制器
        pwm = self.controller.update(cmd, true_state)

        # 3. 执行器
        thrusts, torques = self.actuator.update(pwm)
        F_body, M_body = self.actuator.get_total_wrench(thrusts, torques)

        # 4. 计算张力（用上一步的摆状态，显式耦合）
        F_tether_body = self.pendulum.compute_tension_force()

        # 5. 刚体动力学（RK4 积分，包含张力）
        self.rigid_body.update(F_body, M_body, F_tether_body)

        # 6. 更新摆动力学（用无人机实际 body 加速度）
        # 从刚体获取 body 系加速度
        R_world_body = self.rigid_body._quat_to_rotation_matrix(self.rigid_body.q).T
        accel_world = self.rigid_body._last_accel_ned
        accel_body = R_world_body @ accel_world
        self.pendulum.update(accel_body[0], accel_body[1], self.dt)

        self.time += self.dt

        # 7. 组装输出状态
        true_state = self.rigid_body.get_state()
        true_state['q'] = self.rigid_body.q.copy()
        pendulum_state = self.pendulum.get_state()
        true_state.update(pendulum_state)

        return true_state, pwm

    def run(self, cmd_func, duration, log_interval=0.01):
        """
        cmd_func: callable(time) -> cmd dict
        duration: 总仿真时间 s
        log_interval: 记录间隔 s
        return: logs dict
        """
        n_steps = int(duration / self.dt)
        log_every = max(1, int(log_interval / self.dt))
        logs = {
            'time': [],
            'pos_ned': [],
            'vel_ned': [],
            'roll_deg': [],
            'pitch_deg': [],
            'yaw_deg': [],
            'pwm': [],
            'theta_x_deg': [],
            'theta_y_deg': [],
            'omega_x_dps': [],
            'omega_y_dps': [],
            'tension_n': [],
        }

        for i in range(n_steps):
            cmd = cmd_func(self.time)
            true_state, pwm = self.step(cmd)

            if i % log_every == 0:
                logs['time'].append(self.time)
                logs['pos_ned'].append(true_state['pos_ned'].copy())
                logs['vel_ned'].append(true_state['vel_ned'].copy())
                logs['roll_deg'].append(true_state['roll_deg'])
                logs['pitch_deg'].append(true_state['pitch_deg'])
                logs['yaw_deg'].append(true_state['yaw_deg'])
                logs['pwm'].append(pwm.copy())
                logs['theta_x_deg'].append(true_state['theta_x_deg'])
                logs['theta_y_deg'].append(true_state['theta_y_deg'])
                logs['omega_x_dps'].append(true_state['omega_x_dps'])
                logs['omega_y_dps'].append(true_state['omega_y_dps'])
                logs['tension_n'].append(true_state['tension_n'].copy())

        # 转为 numpy
        for key in logs:
            if key == 'time':
                logs[key] = np.array(logs[key])
            elif key in ['pos_ned', 'vel_ned', 'pwm', 'tension_n']:
                logs[key] = np.array(logs[key])
            else:
                logs[key] = np.array(logs[key])
        return logs
