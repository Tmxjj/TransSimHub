'''
Author: yufei Ji
Date: 2026-04-21
Description: 选择下一个相位 + 动态绿灯时长。
    动作为 (new_phase: int, green_duration: int)，支持从候选集 [10,15,20,25,30,35] 秒中选择绿灯时长。
    黄灯状态机与 choose_next_phase 一致；next_action_time 按实际 green_duration 推算，实现异步决策。
FilePath: TransSimHub/tshub/traffic_light/tls_type/choose_next_phase_with_duration.py
'''
from loguru import logger
from .base_tls import BaseTLS

# 合法绿灯时长候选集（秒）
GREEN_DURATION_CANDIDATES = [10, 15, 20, 25, 30, 35]
MIN_GREEN = GREEN_DURATION_CANDIDATES[0]
MAX_GREEN = GREEN_DURATION_CANDIDATES[-1]


class choose_next_phase_with_duration(BaseTLS):
    """联合决策：选相位 + 动态绿灯时长。

    与 choose_next_phase 的核心区别：
      - set_next_phases 接收 (new_phase, green_duration) 二元组
      - next_action_time = sim_step + green_duration + yellow_time
        即下次决策时间由本次选择的绿灯时长决定，而非固定 delta_time，从而实现异步决策。
    """

    def __init__(self, ts_id, sumo,
                 delta_time: int = 27,
                 yellow_time: int = 3) -> None:
        super().__init__(ts_id, sumo)

        self.delta_time = delta_time    # 兼容父类接口，作为默认绿灯时长
        self.yellow_time = yellow_time  # 黄灯持续时间（秒）

        assert yellow_time > 0, "yellow_time 必须大于 0。"
        assert MIN_GREEN > yellow_time, (
            f"最小绿灯时长 {MIN_GREEN}s 必须大于黄灯时长 {yellow_time}s。"
        )

        self.phase_index = 0                    # 当前绿灯相位索引
        self.time_since_last_phase_change = 0   # 距上次相位切换的仿真步数
        self.is_yellow = False                  # 当前是否处于黄灯过渡阶段
        self.next_action_time = 0               # 下一次决策的仿真时间（秒）

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def set_next_phases(self, new_phase: int, green_duration: int) -> None:
        """设置下一个相位及其绿灯时长。

        Args:
            new_phase (int): 目标绿灯相位索引（0-based，只含绿灯相位）。
            green_duration (int): 绿灯持续时间（秒），应来自 GREEN_DURATION_CANDIDATES。
        """
        new_phase = int(new_phase)
        green_duration = self._clamp_duration(green_duration)

        if self.phase_index == new_phase:
            # 保持当前相位，仅重置绿灯时长
            self.sumo.trafficlight.setPhase(self.id, self.phase_index)
            self.sumo.trafficlight.setPhaseDuration(
                tlsID=self.id, phaseDuration=green_duration
            )
            self.next_action_time = self.sim_step + green_duration + self.yellow_time
            logger.debug(
                'SIM: Time: {}; Keep Phase: {}; Duration: {}s; Next Action: {};'.format(
                    self.sim_step, self.phase_index, green_duration, self.next_action_time
                )
            )
        else:
            # 相位切换：先进入黄灯，再由 update() 切换到绿灯
            yellow_phase_id = self.yellow_dict[(self.phase_index, new_phase)]
            self.sumo.trafficlight.setPhase(self.id, yellow_phase_id)
            logger.debug(
                'SIM: Time: {}; Yellow({}->{}) Phase: {}; Duration: {}s;'.format(
                    self.sim_step, self.phase_index, new_phase,
                    new_phase, green_duration
                )
            )
            self.phase_index = new_phase
            # 黄灯结束后绿灯时长由 setPhaseDuration 在 update() 中设置
            self._pending_green_duration = green_duration
            self.next_action_time = self.sim_step + green_duration + self.yellow_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def update(self) -> None:
        """每仿真步调用，负责黄灯 → 绿灯切换并设置绿灯时长。"""
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            self.sumo.trafficlight.setPhase(self.id, self.phase_index)
            # 设置本次绿灯时长（仅对本次相位生效）
            pending = getattr(self, '_pending_green_duration', self.delta_time)
            self.sumo.trafficlight.setPhaseDuration(
                tlsID=self.id, phaseDuration=pending
            )
            self.is_yellow = False
            logger.debug(
                'SIM: Time {}; Yellow->Green Phase: {}; Duration: {}s;'.format(
                    self.sim_step, self.phase_index, pending
                )
            )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_duration(duration: int) -> int:
        """将时长限制在合法候选集范围内，超界取最近端点。"""
        if duration not in GREEN_DURATION_CANDIDATES:
            clamped = min(GREEN_DURATION_CANDIDATES, key=lambda x: abs(x - duration))
            logger.warning(
                f'SIM: 绿灯时长 {duration}s 不在候选集 {GREEN_DURATION_CANDIDATES} 中，'
                f'自动修正为 {clamped}s。'
            )
            return clamped
        return duration
