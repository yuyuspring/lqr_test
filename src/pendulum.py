"""
吊摆模型

包含两种实现：
- PendulumModel: 当前平面双通道模型，保持原有行为
- Pendulum3DModel: 刚性无质量绳 + 点质量载荷的 3D 方向向量模型

摆角定义（与 xp_16_7 一致）：
- theta_x: 绕 body-x 轴的摆角，左偏为正（对应 gyro_angle_x）
- theta_y: 绕 body-y 轴的摆角，前偏为正（对应 gyro_angle_y）

LQR body-x 通道（前后摆）：直接使用 theta_y
LQR body-y 通道（左右摆）：使用 -theta_x（右偏为正）
"""
import numpy as np


def _normalize(vector, fallback):
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return fallback.copy()
    return vector / norm


def _project_to_tangent(vector, direction):
    return vector - direction * float(np.dot(direction, vector))


class PendulumModel:
    """
    3D 吊摆模型

    双向耦合：
    - 无人机加速度 → 摆运动（通过 computeDerivative）
    - 摆运动 → 张力 → 无人机外力（通过 compute_tension）
    """

    def __init__(self, L=15.0, M_payload=150.0, M_drone=120.0,
                 linear_damping=0.15, drag_coeff=1.0, drag_area=0.5,
                 air_density=1.225):
        self.L = L
        self.M_payload = M_payload
        self.M_drone = M_drone
        self.mu = M_payload / (M_drone + M_payload) if M_payload > 0 else 0.0
        self.g = 9.81
        self.linear_damping = linear_damping
        self.drag_coeff = drag_coeff
        self.drag_area = drag_area
        self.air_density = air_density

        # 摆状态
        self.theta_x = 0.0  # 绕 body-x，左偏为正
        self.omega_x = 0.0
        self.theta_y = 0.0  # 绕 body-y，前偏为正
        self.omega_y = 0.0

    def reset(self, theta_x=0.0, omega_x=0.0, theta_y=0.0, omega_y=0.0):
        self.theta_x = theta_x
        self.omega_x = omega_x
        self.theta_y = theta_y
        self.omega_y = omega_y

    # ------------------------------------------------------------------
    # 动力学（与 StandaloneLqrSimulator.computeDerivative 完全一致）
    # ------------------------------------------------------------------
    def _compute_derivative(self, theta, omega, a1):
        """
        计算单通道摆的状态导数
        theta: 摆角 [rad]
        omega: 角速度 [rad/s]
        a1: 无人机在该通道方向的加速度 [m/s^2]
        返回: dtheta, domega
        """
        if self.mu < 1e-6:
            return 0.0, 0.0

        st = np.sin(theta)
        ct = np.cos(theta)

        denom = 1.0 - self.mu * ct * ct
        if denom < 1e-3:
            denom = 1e-3

        dv = ((1.0 - self.mu) * a1 + self.mu * self.g * st * ct
              + self.mu * self.L * omega * omega * st) / denom

        gravityTorque = self.g * st
        inertialTorque = dv * ct

        dtheta = omega
        domega = -(gravityTorque + inertialTorque) / self.L
        return dtheta, domega

    def _step_rk4(self, theta, omega, a1, dt):
        """RK4 单步积分"""
        k1_t, k1_w = self._compute_derivative(theta, omega, a1)

        t2 = theta + k1_t * dt * 0.5
        w2 = omega + k1_w * dt * 0.5
        k2_t, k2_w = self._compute_derivative(t2, w2, a1)

        t3 = theta + k2_t * dt * 0.5
        w3 = omega + k2_w * dt * 0.5
        k3_t, k3_w = self._compute_derivative(t3, w3, a1)

        t4 = theta + k3_t * dt
        w4 = omega + k3_w * dt
        k4_t, k4_w = self._compute_derivative(t4, w4, a1)

        theta_new = theta + (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t) * dt / 6.0
        omega_new = omega + (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) * dt / 6.0
        return theta_new, omega_new

    def update(self, a_body_x, a_body_y, dt):
        """
        根据无人机 body 系加速度更新摆状态

        a_body_x: body-x 方向加速度（对应前后摆通道）
        a_body_y: body-y 方向加速度（对应左右摆通道）
        dt: 积分步长 [s]
        """
        # body-x 通道（前后摆，绕 body-y）
        self.theta_y, self.omega_y = self._step_rk4(
            self.theta_y, self.omega_y, a_body_x, dt)

        # body-y 通道（左右摆，绕 body-x）
        # 统一约定：theta_x 左偏为正。
        # StandaloneLqrSimulator 的单通道植物对这个通道天然使用“右偏为正”，
        # 因此这里对输入加速度取反，把内部状态语义翻到“左偏为正”。
        self.theta_x, self.omega_x = self._step_rk4(
            self.theta_x, self.omega_x, -a_body_y, dt)

    # ------------------------------------------------------------------
    # 张力计算
    # ------------------------------------------------------------------
    def compute_tension_force(self):
        """
        计算绳索张力对无人机的作用力（body frame）

        返回: F_tether_body [3] (N)
        """
        if self.mu < 1e-6:
            return np.zeros(3)

        theta_mag = np.sqrt(self.theta_x**2 + self.theta_y**2)
        ct = np.cos(theta_mag)

        # 张力大小（静态 + 动态）
        # T = m_p * (g * cos(theta) + L * omega^2)
        # 分别计算两个通道的贡献
        T_x = self.M_payload * (self.g * np.cos(self.theta_y)
                                + self.L * self.omega_y**2)
        T_y = self.M_payload * (self.g * np.cos(self.theta_x)
                                + self.L * self.omega_x**2)
        # 合成张力（取平均值作为保守估计）
        T = 0.5 * (T_x + T_y)
        T = max(T, 0.0)

        # 张力在 body frame 的分量
        # 绳索方向（从无人机指向载荷）：
        #   dx = sin(theta_y), dy = -sin(theta_x), dz = -cos(theta_mag)
        # 张力对无人机的作用力（沿绳索指向载荷）：
        Fx = T * np.sin(self.theta_y)
        Fy = -T * np.sin(self.theta_x)
        Fz = -T * np.cos(theta_mag)

        return np.array([Fx, Fy, Fz])

    def get_lqr_theta_x(self):
        """LQR body-y 通道需要的摆角（右偏为正 = -左偏为正）"""
        return -self.theta_x

    def get_lqr_omega_x(self):
        """LQR body-y 通道需要的角速度"""
        return -self.omega_x

    def get_lqr_theta_y(self):
        """LQR body-x 通道需要的摆角（前偏为正）"""
        return self.theta_y

    def get_lqr_omega_y(self):
        """LQR body-x 通道需要的角速度"""
        return self.omega_y

    def get_state(self):
        """返回摆状态字典"""
        return {
            'theta_x_deg': np.rad2deg(self.theta_x),
            'omega_x_dps': np.rad2deg(self.omega_x),
            'theta_y_deg': np.rad2deg(self.theta_y),
            'omega_y_dps': np.rad2deg(self.omega_y),
            'tension_n': self.compute_tension_force(),
        }


class Pendulum3DModel:
    """
    3D 吊摆模型。

    动力学核心使用与 PendulumModel / StandaloneLqrSimulator 完全相同的
    含 mu 耦合的 RK4 平面摆方程（body-x 与 body-y 两通道独立解耦）。
    内部同时维护方向向量表示（用于张力计算和外部接口），保证大角度下
    无万向节锁死。
    """

    def __init__(self, L=15.0, M_payload=150.0, M_drone=120.0,
                 linear_damping=0.0, drag_coeff=0.0, drag_area=0.0,
                 air_density=0.0, g=9.81):
        self.L = L
        self.M_payload = M_payload
        self.M_drone = M_drone
        self.mu = M_payload / (M_drone + M_payload) if M_payload > 0 else 0.0
        self.g = g
        self.linear_damping = linear_damping
        self.drag_coeff = drag_coeff
        self.drag_area = drag_area
        self.air_density = air_density

        # 方向向量状态（由角度同步，不直接积分）
        self.direction_world = np.array([0.0, 0.0, -1.0], dtype=float)
        self.direction_rate_world = np.zeros(3, dtype=float)
        self.rotation_body_to_world = np.eye(3)
        self.last_uav_accel_world = np.zeros(3, dtype=float)

        # 角度状态（由 RK4 直接积分）
        self.theta_x = 0.0
        self.omega_x = 0.0
        self.theta_y = 0.0
        self.omega_y = 0.0

    def reset(self, theta_x=0.0, omega_x=0.0, theta_y=0.0, omega_y=0.0):
        self.theta_x = theta_x
        self.omega_x = omega_x
        self.theta_y = theta_y
        self.omega_y = omega_y
        self._update_direction_from_angles()
        self._update_angles_from_direction()

    # ------------------------------------------------------------------
    # 动力学（与 PendulumModel._compute_derivative / StandaloneLqrSimulator 完全一致）
    # ------------------------------------------------------------------
    def _compute_derivative(self, theta, omega, a1):
        """
        计算单通道摆的状态导数
        theta: 摆角 [rad]
        omega: 角速度 [rad/s]
        a1: 无人机在该通道方向的加速度 [m/s^2]
        返回: dtheta, domega
        """
        if self.mu < 1e-6:
            return 0.0, 0.0

        st = np.sin(theta)
        ct = np.cos(theta)

        denom = 1.0 - self.mu * ct * ct
        if denom < 1e-3:
            denom = 1e-3

        dv = ((1.0 - self.mu) * a1 + self.mu * self.g * st * ct
              + self.mu * self.L * omega * omega * st) / denom

        gravityTorque = self.g * st
        inertialTorque = dv * ct
        linearDrag = self.linear_damping * omega

        # 二次风阻
        vxPay = 0.0 + self.L * omega * ct  # v_drone 假设为 0（相对速度）
        vzPay = self.L * omega * st
        vAbsSq = vxPay * vxPay + vzPay * vzPay
        vAbs = np.sqrt(vAbsSq)

        quadraticDrag = 0.0
        if vAbs > 1e-6 and self.M_payload > 1e-6:
            dragForceMag = (0.5 * self.air_density * self.drag_coeff
                            * self.drag_area * vAbsSq)
            fDragX = -dragForceMag * vxPay / vAbs
            fDragZ = -dragForceMag * vzPay / vAbs
            fTangential = fDragX * ct + fDragZ * st
            quadraticDrag = fTangential / (self.M_payload * self.L)

        dtheta = omega
        domega = -(gravityTorque + inertialTorque) / self.L - linearDrag + quadraticDrag
        return dtheta, domega

    def _step_rk4(self, theta, omega, a1, dt):
        """RK4 单步积分"""
        k1_t, k1_w = self._compute_derivative(theta, omega, a1)

        t2 = theta + k1_t * dt * 0.5
        w2 = omega + k1_w * dt * 0.5
        k2_t, k2_w = self._compute_derivative(t2, w2, a1)

        t3 = theta + k2_t * dt * 0.5
        w3 = omega + k2_w * dt * 0.5
        k3_t, k3_w = self._compute_derivative(t3, w3, a1)

        t4 = theta + k3_t * dt
        w4 = omega + k3_w * dt
        k4_t, k4_w = self._compute_derivative(t4, w4, a1)

        theta_new = theta + (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t) * dt / 6.0
        omega_new = omega + (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) * dt / 6.0
        return theta_new, omega_new

    # ------------------------------------------------------------------
    # 角度 ↔ 方向向量 同步
    # ------------------------------------------------------------------
    def _update_direction_from_angles(self):
        """由当前 theta_x/theta_y/omega_x/omega_y 精确同步方向向量。"""
        c_x = np.cos(self.theta_x)
        s_x = np.sin(self.theta_x)
        c_y = np.cos(self.theta_y)
        s_y = np.sin(self.theta_y)
        direction_body = np.array([
            s_y * c_x,
            -c_y * s_x,
            -c_y * c_x,
        ], dtype=float)
        self.direction_world = _normalize(
            self.rotation_body_to_world @ direction_body,
            np.array([0.0, 0.0, -1.0], dtype=float),
        )
        rate_body = np.array([
            c_y * c_x * self.omega_y - s_y * s_x * self.omega_x,
            s_y * s_x * self.omega_y - c_y * c_x * self.omega_x,
            s_y * c_x * self.omega_y + c_y * s_x * self.omega_x,
        ], dtype=float)
        self.direction_rate_world = _project_to_tangent(
            self.rotation_body_to_world @ rate_body,
            self.direction_world,
        )

    def _update_angles_from_direction(self):
        direction_body = self.rotation_body_to_world.T @ self.direction_world
        rate_body = self.rotation_body_to_world.T @ self.direction_rate_world
        direction_body = _normalize(direction_body, np.array([0.0, 0.0, -1.0]))

        self.theta_y = float(np.arctan2(direction_body[0], -direction_body[2]))
        self.theta_x = float(np.arctan2(-direction_body[1], -direction_body[2]))

        denom_y = max(direction_body[0] * direction_body[0]
                      + direction_body[2] * direction_body[2], 1e-9)
        self.omega_y = float(((-direction_body[2]) * rate_body[0]
                              + direction_body[0] * rate_body[2]) / denom_y)

        u_x = -direction_body[1]
        v_x = -direction_body[2]
        u_x_dot = -rate_body[1]
        v_x_dot = -rate_body[2]
        denom_x = max(u_x * u_x + v_x * v_x, 1e-9)
        self.omega_x = float((v_x * u_x_dot - u_x * v_x_dot) / denom_x)

    # ------------------------------------------------------------------
    # 主更新
    # ------------------------------------------------------------------
    def update_from_world(self, accel_world, rotation_body_to_world, dt):
        self.rotation_body_to_world = rotation_body_to_world.copy()
        self.last_uav_accel_world = accel_world.copy()
        if self.M_payload < 1e-6:
            self.direction_world = np.array([0.0, 0.0, -1.0], dtype=float)
            self.direction_rate_world = np.zeros(3, dtype=float)
            self._update_angles_from_direction()
            return

        # 将 world 加速度转到 body 系
        accel_body = rotation_body_to_world.T @ accel_world

        # body-x 通道（前后摆，绕 body-y）
        self.theta_y, self.omega_y = self._step_rk4(
            self.theta_y, self.omega_y, accel_body[0], dt)

        # body-y 通道（左右摆，绕 body-x）
        # 统一约定：theta_x 左偏为正，因此这里对 body-y 加速度取反。
        self.theta_x, self.omega_x = self._step_rk4(
            self.theta_x, self.omega_x, -accel_body[1], dt)

        # 同步方向向量
        self._update_direction_from_angles()

    def update(self, a_body_x, a_body_y, dt):
        accel_world = self.rotation_body_to_world @ np.array([a_body_x, a_body_y, 0.0])
        self.update_from_world(accel_world, self.rotation_body_to_world, dt)

    # ------------------------------------------------------------------
    # 张力计算
    # ------------------------------------------------------------------
    def compute_tension_force(self, rotation_body_to_world=None, uav_accel_world=None):
        """
        计算绳索张力对无人机的作用力（body frame）。
        使用与 PendulumModel 完全一致的静态+动态张力公式。
        """
        if self.M_payload < 1e-6:
            return np.zeros(3)
        if rotation_body_to_world is None:
            rotation_body_to_world = self.rotation_body_to_world

        theta_mag = np.sqrt(self.theta_x**2 + self.theta_y**2)

        T_x = self.M_payload * (self.g * np.cos(self.theta_y)
                                + self.L * self.omega_y**2)
        T_y = self.M_payload * (self.g * np.cos(self.theta_x)
                                + self.L * self.omega_x**2)
        T = 0.5 * (T_x + T_y)
        T = max(T, 0.0)

        Fx = T * np.sin(self.theta_y)
        Fy = -T * np.sin(self.theta_x)
        Fz = -T * np.cos(theta_mag)

        force_body = np.array([Fx, Fy, Fz])
        return rotation_body_to_world.T @ force_body

    def get_lqr_theta_x(self):
        return -self.theta_x

    def get_lqr_omega_x(self):
        return -self.omega_x

    def get_lqr_theta_y(self):
        return self.theta_y

    def get_lqr_omega_y(self):
        return self.omega_y

    def get_state(self):
        return {
            'theta_x_deg': np.rad2deg(self.theta_x),
            'omega_x_dps': np.rad2deg(self.omega_x),
            'theta_y_deg': np.rad2deg(self.theta_y),
            'omega_y_dps': np.rad2deg(self.omega_y),
            'tension_n': self.compute_tension_force(),
            'pendulum_direction_world': self.direction_world.copy(),
            'pendulum_direction_rate_world': self.direction_rate_world.copy(),
        }
