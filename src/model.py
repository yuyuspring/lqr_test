"""
共轴八旋翼动力学模型
包含执行器模型（电机一阶+出力折扣）和 6DOF 刚体动力学
"""
import numpy as np


class ActuatorModel:
    """
    8 电机执行器模型
    PWM [0,1000] → 转速（一阶 2Hz）→ 拉力/扭矩
    下层电机（M5~M8）实际拉力为上层的 67%
    """

    # 电机布局参数（与 lqr_test_1 对齐，CCW=+1, CW=-1）
    DIRECTIONS = np.array([1.0, -1.0, 1.0, -1.0,   # 上层 M1~M4
                           -1.0, 1.0, -1.0, 1.0])   # 下层 M5~M8

    # 安装位置（机体坐标系，单位 m）
    # 前x+，右y+，上z+
    POSITIONS = np.array([
        [ 1.0, -1.0, 0.0],   # M1 左前
        [ 1.0,  1.0, 0.0],   # M2 右前
        [-1.0,  1.0, 0.0],   # M3 右后
        [-1.0, -1.0, 0.0],   # M4 左后
        [ 1.0, -1.0, 0.0],   # M5 左前
        [ 1.0,  1.0, 0.0],   # M6 右前
        [-1.0,  1.0, 0.0],   # M7 右后
        [-1.0, -1.0, 0.0],   # M8 左后
    ])

    # 下层折扣
    LOWER_DISCOUNT = np.array([1.0, 1.0, 1.0, 1.0,
                               0.67, 0.67, 0.67, 0.67])

    def __init__(self, dt,
                 max_rpm=1800.0,
                 max_thrust_kg=100.0,
                 max_torque_nm=80.0,
                 bandwidth_hz=2.0):
        self.dt = dt
        self.max_rpm = max_rpm
        self.max_omega = max_rpm * 2.0 * np.pi / 60.0   # rad/s
        self.max_thrust = max_thrust_kg * 9.81           # N
        self.max_torque = max_torque_nm                  # Nm
        self.tau = 1.0 / (2.0 * np.pi * bandwidth_hz)    # 一阶时间常数

        # 拉力系数与扭矩系数（基于最大转速）
        self.k_thrust = self.max_thrust / (self.max_omega ** 2)
        self.k_torque = self.max_torque / (self.max_omega ** 2)

        self.omega = np.zeros(8)

    def compute_hover_pwm(self, total_weight_n):
        """
        计算悬停时各电机的 PWM 值
        total_weight_n: 总重力 (N)
        return: float，单个电机的悬停 PWM（假设均匀分配）
        """
        # 总推力 = 4*T_upper + 4*0.67*T_upper = 6.68*T_upper = total_weight_n
        T_upper = total_weight_n / 6.68
        # 由 T = k * omega^2 得 omega
        omega = np.sqrt(T_upper / self.k_thrust)
        pwm = omega / self.max_omega * 1000.0
        return pwm

    def reset(self):
        self.omega = np.zeros(8)

    def update(self, pwm):
        """
        pwm: np.array shape(8,), range [0, 1000]
        return: thrusts[8] N, torques[8] Nm
        """
        pwm = np.clip(pwm, 0.0, 1000.0)
        omega_cmd = pwm / 1000.0 * self.max_omega

        # 一阶惯性
        domega = (omega_cmd - self.omega) / self.tau
        self.omega += domega * self.dt

        # 拉力与扭矩
        thrusts = self.k_thrust * self.omega ** 2 * self.LOWER_DISCOUNT
        torques = self.k_torque * self.omega ** 2 * self.DIRECTIONS
        return thrusts, torques

    def get_total_wrench(self, thrusts, torques):
        """
        将 8 路拉力和扭矩合成为机体坐标系下的总力和总力矩
        return: F_body [3], M_body [3]
        """
        F_body = np.array([0.0, 0.0, np.sum(thrusts)])
        M_body = np.zeros(3)
        for i in range(8):
            r = self.POSITIONS[i]
            F_i = np.array([0.0, 0.0, thrusts[i]])
            M_body += np.cross(r, F_i)
        M_body[2] += np.sum(torques)
        return F_body, M_body


class RigidBodyModel:
    """
    6DOF 刚体动力学
    状态：pos_ned[3], vel_ned[3], q[4] (机体->NED, q0实部), omega[3] rad/s body
    """

    def __init__(self, mass, inertia_diag, dt, g=9.81):
        self.mass = mass
        self.I = np.diag(inertia_diag)          # kg·m²
        self.I_inv = np.linalg.inv(self.I)
        self.dt = dt
        self.g = g

        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # 初始水平（ENU: world z-up, body z-up 对齐）
        self.omega = np.zeros(3)

    def reset(self, pos=None, vel=None, q=None, omega=None):
        self.pos = pos.copy() if pos is not None else np.zeros(3)
        self.vel = vel.copy() if vel is not None else np.zeros(3)
        self.q = q.copy() if q is not None else np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = omega.copy() if omega is not None else np.zeros(3)

    def get_state(self):
        """返回完整状态字典"""
        # 从四元数提取欧拉角（roll, pitch, yaw）单位 deg
        q0, q1, q2, q3 = self.q
        roll = np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 ** 2 + q2 ** 2))
        pitch = np.arcsin(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 ** 2 + q3 ** 2))

        R_ned_to_body = self._quat_to_rotation_matrix(self.q).T
        accel_body = R_ned_to_body @ self._last_accel_ned if hasattr(self, '_last_accel_ned') else np.zeros(3)
        alpha = getattr(self, '_last_alpha', np.zeros(3))

        return {
            'pos_ned': self.pos.copy(),
            'vel_ned': self.vel.copy(),
            'roll_deg': np.rad2deg(roll),
            'pitch_deg': np.rad2deg(pitch),
            'yaw_deg': np.rad2deg(yaw),
            'p_dps': np.rad2deg(self.omega[0]),
            'q_dps': np.rad2deg(self.omega[1]),
            'r_dps': np.rad2deg(self.omega[2]),
            'alt_m': self.pos[2],
            'vz_up_mps': self.vel[2],
            'az_up_mpss': self._last_accel_ned[2] if hasattr(self, '_last_accel_ned') else 0.0,
            'accel_body': accel_body,
            'omega_radps': self.omega.copy(),
            'alpha': alpha,
        }

    def update(self, F_body, M_body, F_external_body=None):
        """
        使用 RK4 积分一步
        F_body: 机体坐标系总力 [3]（推力）
        M_body: 机体坐标系总力矩 [3]
        F_external_body: 额外外力 [3]（如绳索张力）
        """
        if F_external_body is None:
            F_external_body = np.zeros(3)
        dt = self.dt
        state = np.concatenate([self.pos, self.vel, self.q, self.omega])

        def dynamics(y):
            pos = y[0:3]
            vel = y[3:6]
            q = y[6:10]
            omega = y[10:13]

            # 归一化四元数（防止数值漂移）
            q = q / np.linalg.norm(q)

            R_world_body = self._quat_to_rotation_matrix(q)
            # ENU 系：world z-up，body z-up，推力 body [0,0,T] 直接映射到 world [0,0,T]
            F_total_body = F_body + F_external_body
            F_world = R_world_body @ F_total_body + np.array([0.0, 0.0, -self.mass * self.g])
            accel = F_world / self.mass

            # 四元数微分
            qdot = self._quat_derivative(q, omega)

            # 角加速度
            alpha = self.I_inv @ (M_body - np.cross(omega, self.I @ omega))

            return np.concatenate([vel, accel, qdot, alpha])

        # RK4
        k1 = dynamics(state)
        k2 = dynamics(state + 0.5 * dt * k1)
        k3 = dynamics(state + 0.5 * dt * k2)
        k4 = dynamics(state + dt * k3)

        state_new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.pos = state_new[0:3]
        self.vel = state_new[3:6]
        self.q = state_new[6:10]
        self.q /= np.linalg.norm(self.q)
        self.omega = state_new[10:13]

        # 保存上一时刻的线加速度和角加速度，供传感器模型使用
        R_world_body = self._quat_to_rotation_matrix(self.q)
        F_total_body = F_body + F_external_body
        F_world = R_world_body @ F_total_body + np.array([0.0, 0.0, -self.mass * self.g])
        self._last_accel_ned = F_world / self.mass
        self._last_alpha = self.I_inv @ (M_body - np.cross(self.omega, self.I @ self.omega))

    @staticmethod
    def _quat_to_rotation_matrix(q):
        """四元数（机体->NED）转旋转矩阵 R_body_to_ned"""
        q0, q1, q2, q3 = q
        R = np.array([
            [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)],
        ])
        return R

    @staticmethod
    def _quat_derivative(q, omega):
        """四元数导数 dq/dt = 0.5 * q ⊗ [0, omega]"""
        q0, q1, q2, q3 = q
        p, q_, r = omega
        return 0.5 * np.array([
            -q1 * p - q2 * q_ - q3 * r,
             q0 * p + q2 * r - q3 * q_,
             q0 * q_ - q1 * r + q3 * p,
             q0 * r + q1 * q_ - q2 * p,
        ])
