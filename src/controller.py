"""
共轴八旋翼控制器
严格依据 controller.md 实现全部控制通道

新增：支持 PVA / VA 两种轨迹跟踪模式
- pva: 位置→速度→加速度→姿态（原有模式）
- va:  速度→加速度→姿态（LQR 输出 vel/acc 时使用）
"""
import numpy as np


class PID:
    """连续时间 PID，带积分限幅和输出限幅。"""

    def __init__(self, Kp, Ki, Kd, dt, out_limit=None, int_limit=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.out_limit = out_limit
        self.int_limit = int_limit
        self.integral = 0.0
        self.prev_err = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0

    def update(self, err):
        self.integral += err * self.dt
        if self.int_limit is not None:
            self.integral = np.clip(self.integral, -self.int_limit, self.int_limit)
        derivative = (err - self.prev_err) / self.dt
        self.prev_err = err
        out = self.Kp * err + self.Ki * self.integral + self.Kd * derivative
        if self.out_limit is not None:
            out = np.clip(out, -self.out_limit, self.out_limit)
        return out


class LeadLag:
    """
    超前-滞后补偿器：G(s) = (1 + a*s) / (1 + b*s)
    状态空间实现：tau = b, K = a/b
    dx = (u - x) / tau
    y = (1 - K)*x + K*u
    """

    def __init__(self, a, b, dt):
        self.a = a
        self.b = b
        self.dt = dt
        self.tau = b
        self.K = a / b if b != 0 else 1.0
        self.x = 0.0

    def reset(self):
        self.x = 0.0

    def update(self, u):
        dx = (u - self.x) / self.tau
        self.x += dx * self.dt
        y = (1.0 - self.K) * self.x + self.K * u
        return y


class Controller:
    """
    全状态控制器

    输入指令：dict，支持两种模式
    模式 'pva' (默认):
        cmd = {'pos_ned': np.array shape(2,), 'yaw': float deg, 'alt': float m}
    模式 'va':
        cmd = {'vel_ned': np.array shape(2,), 'acc_ned': np.array shape(2,),
               'yaw': float deg, 'alt': float m}

    输入状态：dict，包含 pos_ned, vel_ned, roll[deg], pitch[deg], yaw[deg],
                     p[deg/s], q[deg/s], r[deg/s], alt[m], v_z_up[m/s], a_z_up[m/ss]
    输出：8 路 PWM [0, 1000]
    """

    def __init__(self, dt, hover_throttle=0.0, trajectory_mode='pva'):
        self.dt = dt
        self.hover_throttle = hover_throttle
        self.trajectory_mode = trajectory_mode  # 'pva' or 'va'

        # 水平位置环（仅在 pva 模式使用）
        self.pid_px = PID(1.0, 0.0, 0.0, dt, out_limit=5.0)
        self.pid_py = PID(1.0, 0.0, 0.0, dt, out_limit=5.0)

        # 水平速度环
        self.pid_vx = PID(1.0, 0.02, 0.0, dt, out_limit=5.0)
        self.pid_vy = PID(1.0, 0.02, 0.0, dt, out_limit=5.0)

        # 高度环
        self.pid_alt = PID(0.7, 0.0, 0.0, dt, out_limit=5.0)

        # 垂直速度环
        self.pid_vz = PID(2.0, 0.0, 0.0, dt, out_limit=5.0)

        # 垂直加速度环
        self.pid_az = PID(0.1, 2.0, 0.0, dt, int_limit=3.0)

        # 滚转角
        self.pid_roll = PID(5.0, 0.0, 0.0, dt, out_limit=60.0)

        # 滚转角速度环 + Lead-Lag
        a_lead = 1.0 / (2.0 * np.pi * 2.0)
        b_lag = 1.0 / (2.0 * np.pi * 20.0)
        self.pid_p = PID(0.1, 0.05, 0.0, dt, out_limit=300.0)
        self.leadlag_p = LeadLag(a_lead, b_lag, dt)
        self.gain_roll = 100.0 / 7.0

        # 俯仰角
        self.pid_pitch = PID(5.0, 0.0, 0.0, dt, out_limit=60.0)

        # 俯仰角速度环 + Lead-Lag
        self.pid_q = PID(0.1, 0.05, 0.0, dt, out_limit=300.0)
        self.leadlag_q = LeadLag(a_lead, b_lag, dt)
        self.gain_pitch = 100.0 / 7.0

        # 偏航角
        self.pid_yaw = PID(2.0, 0.0, 0.0, dt, out_limit=45.0)

        # 偏航角速度环
        self.pid_r = PID(0.3, 0.1, 0.0, dt, out_limit=250.0)
        self.gain_yaw = 100.0 / 10.0

    def reset(self):
        for attr in dir(self):
            if attr.startswith('pid_') or attr.startswith('leadlag_'):
                getattr(self, attr).reset()

    def update(self, cmd, state):
        """
        cmd: dict
            pva 模式: {'pos_ned': np.array(2,), 'yaw': float, 'alt': float}
            va  模式: {'vel_ned': np.array(2,), 'acc_ned': np.array(2,),
                       'yaw': float, 'alt': float}
        state: dict
        """
        # ---------- 水平通道 ----------
        if self.trajectory_mode == 'va':
            # VA 模式：直接跟踪 LQR 输出的 vel_ref + acc_ref
            vel_ref = cmd.get('vel_ned', np.zeros(2))
            acc_ref = cmd.get('acc_ned', np.zeros(2))
            vx_cmd = vel_ref[0]
            vy_cmd = vel_ref[1]
            # 速度环用 vel_ref 作为指令，acc_ref 作为前馈
            vel_err = np.array([vx_cmd, vy_cmd]) - state['vel_ned'][:2]
            ax_cmd_world = self.pid_vx.update(vel_err[0]) + acc_ref[0]
            ay_cmd_world = self.pid_vy.update(vel_err[1]) + acc_ref[1]
        else:
            # PVA 模式：原有级联 PID
            pos_err = cmd['pos_ned'] - state['pos_ned'][:2]
            vx_cmd = self.pid_px.update(pos_err[0])
            vy_cmd = self.pid_py.update(pos_err[1])

            vel_err = np.array([vx_cmd, vy_cmd]) - state['vel_ned'][:2]
            ax_cmd_world = self.pid_vx.update(vel_err[0])
            ay_cmd_world = self.pid_vy.update(vel_err[1])

        # 加速度指令转到机体坐标系
        yaw_rad = np.deg2rad(state['yaw_deg'])
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)
        ax_body = ax_cmd_world * cos_y + ay_cmd_world * sin_y
        ay_body = -ax_cmd_world * sin_y + ay_cmd_world * cos_y

        g = 9.81
        roll_cmd = np.rad2deg(np.clip(-ay_body / g, -1.0, 1.0))
        pitch_cmd = np.rad2deg(np.clip(ax_body / g, -1.0, 1.0))

        # ---------- 高度/油门通道 ----------
        alt_err = cmd['alt'] - state['alt_m']
        vz_cmd = self.pid_alt.update(alt_err)
        vz_err = vz_cmd - state['vz_up_mps']
        az_cmd = self.pid_vz.update(vz_err)
        az_err = az_cmd - state['az_up_mpss']
        thro_raw = 100.0 * self.pid_az.update(az_err)

        roll_rad = np.deg2rad(state['roll_deg'])
        pitch_rad = np.deg2rad(state['pitch_deg'])
        cos_r = np.cos(roll_rad)
        cos_p = np.cos(pitch_rad)
        tilt_comp = max(cos_r * cos_p, 0.2)
        servo_thro = self.hover_throttle + thro_raw / tilt_comp

        # ---------- 滚转通道 ----------
        roll_err = roll_cmd - state['roll_deg']
        p_cmd = self.pid_roll.update(roll_err)
        p_err = p_cmd - state['p_dps']
        p_out = self.pid_p.update(p_err)
        p_ll = self.leadlag_p.update(p_out)
        servo_roll = self.gain_roll * p_ll

        # ---------- 俯仰通道 ----------
        pitch_err = pitch_cmd - state['pitch_deg']
        q_cmd = self.pid_pitch.update(pitch_err)
        q_err = q_cmd - state['q_dps']
        q_out = self.pid_q.update(q_err)
        q_ll = self.leadlag_q.update(q_out)
        servo_pitch = -self.gain_pitch * q_ll

        # ---------- 偏航通道 ----------
        yaw_err = self._normalize_angle(cmd['yaw'] - state['yaw_deg'])
        r_cmd = self.pid_yaw.update(yaw_err)
        r_err = r_cmd - state['r_dps']
        r_out = self.pid_r.update(r_err)
        servo_yaw = -self.gain_yaw * r_out

        # ---------- 控制分配 ----------
        st = servo_thro
        sr = servo_roll
        sp = servo_pitch
        sy = servo_yaw

        pwm = np.array([
            st - sr + sp - sy,
            st + sr + sp + sy,
            st + sr - sp - sy,
            st - sr - sp + sy,
            st - sr + sp + sy,
            st + sr + sp - sy,
            st + sr - sp + sy,
            st - sr - sp - sy,
        ])

        pwm = np.clip(pwm, 0.0, 1000.0)
        return pwm

    @staticmethod
    def _normalize_angle(angle_deg):
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg
