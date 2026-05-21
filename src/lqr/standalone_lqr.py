"""
Standalone LQR Simulator
Python port of xp_16_7/src/app/planner/lib/lqr/standalone_lqr.cpp

一维吊摆 LQR 仿真器：
- 完整非线性 RK4 动力学
- LQI 状态反馈 + Scheme B 精确非线性前馈
- jerk limit + anti-windup
- 2D 增益查表（mu x L）+ 双线性插值
"""
import numpy as np
from .gain_table import K_DEFAULT_GAIN_TABLE, K_MU_TABLE, K_L_TABLE, K_MU_COUNT, K_L_COUNT


class State:
    """摆动力学状态 [p, v, theta, omega] + 时间 + 加速度"""
    def __init__(self):
        self.time = 0.0
        self.p = 0.0
        self.v = 0.0
        self.a = 0.0
        self.theta = 0.0
        self.omega = 0.0


class StepMetrics:
    """单步调试指标"""
    def __init__(self):
        self.currentVel = 0.0
        self.vRef = 0.0
        self.err = 0.0
        self.theta = 0.0
        self.omega = 0.0
        self.axRef = 0.0
        self.feedforwardAcc = 0.0
        self.swingUavAcc = 0.0
        self.integralBefore = 0.0
        self.integralPreview = 0.0
        self.integralAfter = 0.0
        self.antiWindupGain = 0.0
        self.antiWindupDelta = 0.0
        self.antiWindupCorrection = 0.0
        self.antiWindupPreviewAcc = 0.0
        self.antiWindupPreviewSatAcc = 0.0
        self.targetAcc = 0.0
        self.cmdAcc = 0.0


class Config:
    """LQR 配置参数"""
    def __init__(self):
        self.dt = 0.02
        self.ropeLength = 15.0
        self.lqrAxMax = 5.0
        self.lqrJerkMax = 10.0
        self.payloadMass = 150.0
        self.droneMass = 120.0
        self.linearDampingCoeff = 0.15
        self.dragCoeff = 1.0
        self.dragArea = 0.5
        self.airDensity = 1.225
        self.initialTheta = 0.0
        self.initialOmega = 0.0
        self.initialP = 0.0
        self.initialV = 0.0


class StandaloneLqrSimulator:
    """
    单通道 LQR 消摆仿真器

    坐标约定（与 xp_16_7 一致）：
    - body-x 通道：theta 描述前后摆（绕 body-y），前偏为正
    - body-y 通道：theta 描述左右摆（绕 body-x），右偏为正
    """
    kGravity = 9.81

    def __init__(self, config=None):
        if config is None:
            config = Config()
        self.config_ = config
        self.pendulumGain_ = config.payloadMass / (config.droneMass + config.payloadMass)
        self.gainTable_ = K_DEFAULT_GAIN_TABLE.copy()

        # LQI 增益
        self.kV_ = 0.0
        self.kTheta_ = 0.0
        self.kOmega_ = 0.0
        self.kIntegral_ = 0.0

        # 控制器内部状态
        self.integral_ = 0.0
        self.axCmd_ = 0.0
        self.axRef_ = 0.0

        # 物理状态
        self.state_ = State()
        self.lastStepMetrics_ = StepMetrics()

        # 调试日志
        self.debugOs_ = None
        self.debugBatchId_ = 0
        self.debugStepIdx_ = 0

        self.recomputeGains()
        self.resetAll(config.initialP, config.initialV,
                      config.initialTheta, config.initialOmega)

    # ------------------------------------------------------------------
    # 增益查表
    # ------------------------------------------------------------------
    def bilinearInterpolateGain(self, mu, ropeLength):
        muClamped = max(K_MU_TABLE[0], min(K_MU_TABLE[-1], mu))
        lClamped = max(K_L_TABLE[0], min(K_L_TABLE[-1], ropeLength))

        muIdx = 0
        for i in range(K_MU_COUNT - 1):
            if K_MU_TABLE[i] <= muClamped <= K_MU_TABLE[i + 1]:
                muIdx = i
                break
        lIdx = 0
        for i in range(K_L_COUNT - 1):
            if K_L_TABLE[i] <= lClamped <= K_L_TABLE[i + 1]:
                lIdx = i
                break

        tMu = (muClamped - K_MU_TABLE[muIdx]) / (K_MU_TABLE[muIdx + 1] - K_MU_TABLE[muIdx])
        tL = (lClamped - K_L_TABLE[lIdx]) / (K_L_TABLE[lIdx + 1] - K_L_TABLE[lIdx])

        def interp(member_idx):
            f00 = self.gainTable_[muIdx, lIdx, member_idx]
            f10 = self.gainTable_[muIdx + 1, lIdx, member_idx]
            f01 = self.gainTable_[muIdx, lIdx + 1, member_idx]
            f11 = self.gainTable_[muIdx + 1, lIdx + 1, member_idx]
            return ((1.0 - tMu) * (1.0 - tL) * f00
                    + tMu * (1.0 - tL) * f10
                    + (1.0 - tMu) * tL * f01
                    + tMu * tL * f11)

        kV = interp(1)
        kTheta = interp(2)
        kOmega = interp(3)
        kIntegral = interp(4)
        return kV, kTheta, kOmega, kIntegral

    def recomputeGains(self):
        self.kV_, self.kTheta_, self.kOmega_, self.kIntegral_ = \
            self.bilinearInterpolateGain(self.pendulumGain_, self.config_.ropeLength)

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------
    def setPhysicalState(self, p, v, theta, omega):
        self.state_.p = p
        self.state_.v = v
        noPayload = (self.pendulumGain_ < 1e-6)
        self.state_.theta = 0.0 if noPayload else theta
        self.state_.omega = 0.0 if noPayload else omega

    def resetAll(self, p, v, theta, omega):
        self.state_.p = p
        self.state_.v = v
        noPayload = (self.pendulumGain_ < 1e-6)
        self.state_.theta = 0.0 if noPayload else theta
        self.state_.omega = 0.0 if noPayload else omega
        self.state_.a = 0.0
        self.state_.time = 0.0
        self.integral_ = 0.0
        self.axCmd_ = 0.0
        self.axRef_ = 0.0

    def setRopeLength(self, rope):
        if abs(rope - self.config_.ropeLength) < 0.5:
            return
        self.config_.ropeLength = rope
        self.recomputeGains()

    def setMass(self, payloadMass, droneMass):
        if payloadMass < 10.0:
            payloadMass = 0.0
        if (abs(payloadMass - self.config_.payloadMass) < 10.0 and
                abs(droneMass - self.config_.droneMass) < 10.0):
            return
        self.config_.payloadMass = payloadMass
        self.config_.droneMass = droneMass
        self.pendulumGain_ = payloadMass / (droneMass + payloadMass)
        self.recomputeGains()

    # ------------------------------------------------------------------
    # 动力学核心
    # ------------------------------------------------------------------
    def computeDerivative(self, s, a1):
        """
        计算状态导数
        s: 当前状态
        a1: 无人机加速度（控制输入）
        返回: dp, dv, dth, dw
        """
        mu = self.pendulumGain_

        if mu < 1e-6:
            return s.v, a1, 0.0, 0.0

        st = np.sin(s.theta)
        ct = np.cos(s.theta)

        denom = 1.0 - mu * ct * ct
        if denom < 1e-3:
            denom = 1e-3

        dv = ((1.0 - mu) * a1 + mu * self.kGravity * st * ct
              + mu * self.config_.ropeLength * s.omega * s.omega * st) / denom

        dp = s.v
        dth = s.omega

        gravityTorque = self.kGravity * st
        inertialTorque = dv * ct
        linearDrag = self.config_.linearDampingCoeff * s.omega

        # 二次风阻
        vxPay = s.v + self.config_.ropeLength * s.omega * ct
        vzPay = self.config_.ropeLength * s.omega * st
        vAbsSq = vxPay * vxPay + vzPay * vzPay
        vAbs = np.sqrt(vAbsSq)

        quadraticDrag = 0.0
        if vAbs > 1e-6 and self.config_.payloadMass > 1e-6:
            dragForceMag = (0.5 * self.config_.airDensity * self.config_.dragCoeff
                            * self.config_.dragArea * vAbsSq)
            fDragX = -dragForceMag * vxPay / vAbs
            fDragZ = -dragForceMag * vzPay / vAbs
            fTangential = fDragX * ct + fDragZ * st
            quadraticDrag = fTangential / (self.config_.payloadMass * self.config_.ropeLength)

        dw = -(gravityTorque + inertialTorque) / self.config_.ropeLength - linearDrag + quadraticDrag
        return dp, dv, dth, dw

    def stepRK4(self, s, a1, dt):
        """RK4 积分一步"""
        k1_p, k1_v, k1_t, k1_w = self.computeDerivative(s, a1)

        s2 = State()
        s2.p = s.p + k1_p * dt * 0.5
        s2.v = s.v + k1_v * dt * 0.5
        s2.theta = s.theta + k1_t * dt * 0.5
        s2.omega = s.omega + k1_w * dt * 0.5
        k2_p, k2_v, k2_t, k2_w = self.computeDerivative(s2, a1)

        s3 = State()
        s3.p = s.p + k2_p * dt * 0.5
        s3.v = s.v + k2_v * dt * 0.5
        s3.theta = s.theta + k2_t * dt * 0.5
        s3.omega = s.omega + k2_w * dt * 0.5
        k3_p, k3_v, k3_t, k3_w = self.computeDerivative(s3, a1)

        s4 = State()
        s4.p = s.p + k3_p * dt
        s4.v = s.v + k3_v * dt
        s4.theta = s.theta + k3_t * dt
        s4.omega = s.omega + k3_w * dt
        k4_p, k4_v, k4_t, k4_w = self.computeDerivative(s4, a1)

        s.p += (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * dt / 6.0
        s.v += (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * dt / 6.0
        s.theta += (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t) * dt / 6.0
        s.omega += (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) * dt / 6.0
        s.time += dt

        _, dv_tmp, _, _ = self.computeDerivative(s, a1)
        s.a = dv_tmp

    # ------------------------------------------------------------------
    # 前馈
    # ------------------------------------------------------------------
    def computeFeedforward(self, guideA, theta, omega):
        """Scheme B: 精确非线性前馈"""
        if self.pendulumGain_ < 1e-6:
            return guideA
        mu = self.pendulumGain_
        s = np.sin(theta)
        c = np.cos(theta)
        denom = 1.0 - mu * c * c
        if denom < 1e-3:
            return guideA
        numerator = (guideA * denom
                     - mu * self.kGravity * s * c
                     - mu * self.config_.ropeLength * omega * omega * s)
        oneMinusMu = 1.0 - mu
        if oneMinusMu < 1e-3:
            return guideA
        return numerator / oneMinusMu

    # ------------------------------------------------------------------
    # 控制律
    # ------------------------------------------------------------------
    def computeControl(self, v, theta, omega, vRef):
        err = v - vRef
        integral_before = self.integral_

        noPayload = (self.pendulumGain_ < 1e-6)
        ctrl_theta = 0.0 if noPayload else theta
        ctrl_omega = 0.0 if noPayload else omega

        # Scheme B 前馈
        a1_ff = self.computeFeedforward(self.axRef_, ctrl_theta, ctrl_omega)

        # 摆对 UAV 的被动反作用加速度
        swingUavAcc = 0.0
        if not noPayload:
            s = np.sin(ctrl_theta)
            c = np.cos(ctrl_theta)
            denom = 1.0 - self.pendulumGain_ * c * c
            if denom < 1e-3:
                denom = 1e-3
            swingUavAcc = ((self.pendulumGain_ * self.kGravity * s * c
                            + self.pendulumGain_ * self.config_.ropeLength
                            * ctrl_omega * ctrl_omega * s) / denom)

        kI_eff = self.kIntegral_
        kIntegralRate = 1.0
        kIntMax = 3.0

        # 预览积分
        intPreview = self.integral_ + kIntegralRate * err * self.config_.dt
        uFeedbackPreview = -(self.kV_ * err + self.kTheta_ * ctrl_theta
                             + self.kOmega_ * ctrl_omega + kI_eff * intPreview)
        uPreview = a1_ff + uFeedbackPreview
        u_sat_preview = np.clip(uPreview, -self.config_.lqrAxMax, self.config_.lqrAxMax)
        delta = uPreview - u_sat_preview
        k_aw = (self.kV_ > 1e-6) * (kI_eff / self.kV_)
        antiWindupCorrection = k_aw * delta * self.config_.dt

        self.integral_ = intPreview + antiWindupCorrection
        self.integral_ = np.clip(self.integral_, -kIntMax, kIntMax)

        # 最终控制量
        velFeedback = -self.kV_ * err
        pendulumAcc = -(self.kTheta_ * ctrl_theta + self.kOmega_ * ctrl_omega)
        intFeedback = -kI_eff * self.integral_
        uFeedback = velFeedback + pendulumAcc + intFeedback
        u = a1_ff + uFeedback
        u_sat = np.clip(u, -self.config_.lqrAxMax, self.config_.lqrAxMax)

        self.lastStepMetrics_.currentVel = v
        self.lastStepMetrics_.vRef = vRef
        self.lastStepMetrics_.err = err
        self.lastStepMetrics_.theta = ctrl_theta
        self.lastStepMetrics_.omega = ctrl_omega
        self.lastStepMetrics_.axRef = self.axRef_
        self.lastStepMetrics_.feedforwardAcc = a1_ff
        self.lastStepMetrics_.swingUavAcc = swingUavAcc
        self.lastStepMetrics_.integralBefore = integral_before
        self.lastStepMetrics_.integralPreview = intPreview
        self.lastStepMetrics_.integralAfter = self.integral_
        self.lastStepMetrics_.antiWindupGain = k_aw
        self.lastStepMetrics_.antiWindupDelta = delta
        self.lastStepMetrics_.antiWindupCorrection = antiWindupCorrection
        self.lastStepMetrics_.antiWindupPreviewAcc = uPreview
        self.lastStepMetrics_.antiWindupPreviewSatAcc = u_sat_preview
        self.lastStepMetrics_.targetAcc = u_sat
        return u_sat

    def applyJerkLimit(self, target):
        axCmd_before = self.axCmd_
        jerk = (target - axCmd_before) / self.config_.dt
        if jerk > self.config_.lqrJerkMax:
            self.axCmd_ += self.config_.lqrJerkMax * self.config_.dt
        elif jerk < -self.config_.lqrJerkMax:
            self.axCmd_ -= self.config_.lqrJerkMax * self.config_.dt
        else:
            self.axCmd_ = target
        self.lastStepMetrics_.cmdAcc = self.axCmd_
        return self.axCmd_

    def step(self, vRef):
        """单步 LQR：计算控制 → jerk limit → RK4 推进"""
        axTarget = self.computeControl(self.state_.v, self.state_.theta,
                                       self.state_.omega, vRef)
        axCmd = self.applyJerkLimit(axTarget)
        self.stepRK4(self.state_, axCmd, self.config_.dt)

    # ------------------------------------------------------------------
    # Getters / Setters
    # ------------------------------------------------------------------
    def getP(self): return self.state_.p
    def getV(self): return self.state_.v
    def getA(self): return self.state_.a
    def getAxCmd(self): return self.axCmd_
    def getIntegral(self): return self.integral_
    def getTheta(self): return self.state_.theta
    def getOmega(self): return self.state_.omega
    def getKV(self): return self.kV_
    def getKTheta(self): return self.kTheta_
    def getKOmega(self): return self.kOmega_
    def getKIntegral(self): return self.kIntegral_
    def setIntegral(self, val): self.integral_ = val
    def setAxCmd(self, val): self.axCmd_ = val
    def setAxRef(self, val): self.axRef_ = val
    def getAxRef(self): return self.axRef_
    def getLastStepMetrics(self): return self.lastStepMetrics_
    def getConfig(self): return self.config_
    def getPendulumGain(self): return self.pendulumGain_
