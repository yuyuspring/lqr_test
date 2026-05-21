"""
LqrSwingController
Python port of xp_16_7/src/app/planner/src/pilot/lqr_swing_controller.cpp

双通道 LQR 消摆控制器封装（机体系）
- body-x 通道：前后摆 → body-x 加速度
- body-y 通道：左右摆 → body-y 加速度
"""
import numpy as np
from .standalone_lqr import StandaloneLqrSimulator, Config as LqrConfig


class LqrSwingController:
    """
    双通道 LQR 消摆控制器
    """
    kLookaheadSteps = 0
    kExecutedSteps = 5
    kStickReleaseVRefThresh = 0.05
    kStickReleaseVThresh = 0.5

    def __init__(self, dt=0.02, ropeLength=15.0, payloadMass=150.0,
                 droneMass=120.0, lqrAxMax=5.0, lqrJerkMax=10.0):
        cfg = LqrConfig()
        cfg.dt = dt
        cfg.ropeLength = ropeLength
        cfg.payloadMass = payloadMass
        cfg.droneMass = droneMass
        cfg.lqrAxMax = lqrAxMax
        cfg.lqrJerkMax = lqrJerkMax
        self.lqrX_ = StandaloneLqrSimulator(cfg)
        self.lqrY_ = StandaloneLqrSimulator(cfg)

    def updateParams(self, ropeLength, payloadMass, droneMass):
        self.lqrX_.setRopeLength(ropeLength)
        self.lqrX_.setMass(payloadMass, droneMass)
        self.lqrY_.setRopeLength(ropeLength)
        self.lqrY_.setMass(payloadMass, droneMass)

    def processBatch(self,
                     vxRefBody, vyRefBody,
                     axRefBody, ayRefBody,
                     px0_body, vx0_body, thetaBodyX0, omegaBodyX0,
                     py0_body, vy0_body, thetaBodyY0, omegaBodyY0):
        """
        批量处理一段轨迹的速度参考序列（机体系）

        返回: dict with keys:
            'vxOut', 'vyOut', 'axOut', 'ayOut': np.array shape(executedSteps,)
            'debug': list of dict, each with x/y StepMetrics for each step
        """
        N = len(vxRefBody)
        if N == 0 or len(vyRefBody) != N:
            return {
                'vxOut': np.array([]), 'vyOut': np.array([]),
                'axOut': np.array([]), 'ayOut': np.array([]),
                'debug': []
            }

        executedSteps = min(N, self.kExecutedSteps)

        # Shift vRef with lookahead (if any)
        vxRefShifted = np.array(vxRefBody)
        vyRefShifted = np.array(vyRefBody)
        axRefShifted = np.array(axRefBody) if axRefBody is not None else np.array([])
        ayRefShifted = np.array(ayRefBody) if ayRefBody is not None else np.array([])

        if N > self.kLookaheadSteps and self.kLookaheadSteps > 0:
            for i in range(N - self.kLookaheadSteps):
                vxRefShifted[i] = vxRefShifted[i + self.kLookaheadSteps]
                vyRefShifted[i] = vyRefShifted[i + self.kLookaheadSteps]
            for i in range(N - self.kLookaheadSteps, N):
                vxRefShifted[i] = vxRefShifted[N - self.kLookaheadSteps - 1]
                vyRefShifted[i] = vyRefShifted[N - self.kLookaheadSteps - 1]
            if len(axRefShifted) > 0:
                for i in range(N - self.kLookaheadSteps):
                    axRefShifted[i] = axRefShifted[i + self.kLookaheadSteps]
                    ayRefShifted[i] = ayRefShifted[i + self.kLookaheadSteps]
                for i in range(N - self.kLookaheadSteps, N):
                    axRefShifted[i] = axRefShifted[N - self.kLookaheadSteps - 1]
                    ayRefShifted[i] = ayRefShifted[N - self.kLookaheadSteps - 1]

        # 注入初始物理状态
        self.lqrX_.setPhysicalState(px0_body, vx0_body, thetaBodyX0, omegaBodyX0)
        self.lqrY_.setPhysicalState(py0_body, vy0_body, thetaBodyY0, omegaBodyY0)

        # 松杆检测
        stickReleasedX = (len(vxRefBody) > 0
                          and abs(vxRefBody[0]) < self.kStickReleaseVRefThresh
                          and abs(vx0_body) < self.kStickReleaseVThresh)
        stickReleasedY = (len(vyRefBody) > 0
                          and abs(vyRefBody[0]) < self.kStickReleaseVRefThresh
                          and abs(vy0_body) < self.kStickReleaseVThresh)

        savedAxCmdX = 0.0
        savedIntX = 0.0
        savedAxCmdY = 0.0
        savedIntY = 0.0

        vxOut = np.zeros(N)
        vyOut = np.zeros(N)
        axOut = np.zeros(N)
        ayOut = np.zeros(N)
        debug = []

        for i in range(N):
            # 在 executedSteps 处保存 controller 状态
            if i == self.kExecutedSteps:
                savedAxCmdX = self.lqrX_.getAxCmd()
                savedIntX = self.lqrX_.getIntegral()
                savedAxCmdY = self.lqrY_.getAxCmd()
                savedIntY = self.lqrY_.getIntegral()

            # 设置前馈加速度参考
            if i < len(axRefShifted):
                self.lqrX_.setAxRef(0.0 if stickReleasedX else axRefShifted[i])
            if i < len(ayRefShifted):
                self.lqrY_.setAxRef(0.0 if stickReleasedY else ayRefShifted[i])

            self.lqrX_.step(vxRefShifted[i])
            self.lqrY_.step(vyRefShifted[i])

            vxOut[i] = self.lqrX_.getV()
            axOut[i] = self.lqrX_.getAxCmd()
            vyOut[i] = self.lqrY_.getV()
            ayOut[i] = self.lqrY_.getAxCmd()

            # 收集调试信息
            mx = self.lqrX_.getLastStepMetrics()
            my = self.lqrY_.getLastStepMetrics()
            debug.append({
                'x': mx, 'y': my,
                'vxRef': vxRefShifted[i],
                'vyRef': vyRefShifted[i],
            })

        # 如果 batch 少于 executedSteps，从末尾保存
        if N <= self.kExecutedSteps:
            savedAxCmdX = self.lqrX_.getAxCmd()
            savedIntX = self.lqrX_.getIntegral()
            savedAxCmdY = self.lqrY_.getAxCmd()
            savedIntY = self.lqrY_.getIntegral()

        # 恢复 controller 状态到 executed 结束处
        self.lqrX_.setAxCmd(savedAxCmdX)
        self.lqrX_.setIntegral(savedIntX)
        self.lqrY_.setAxCmd(savedAxCmdY)
        self.lqrY_.setIntegral(savedIntY)

        return {
            'vxOut': vxOut[:executedSteps],
            'vyOut': vyOut[:executedSteps],
            'axOut': axOut[:executedSteps],
            'ayOut': ayOut[:executedSteps],
            'debug': debug[:executedSteps],
        }
