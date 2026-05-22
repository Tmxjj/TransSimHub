'''
@Author: WANG Maonan
@Date: 2024-07-19 23:20:18
@Description: 这里的相机位置是固定不动的, 也就是摄像头
LastEditTime: 2026-05-22 22:00:00
'''
import math
import numpy as np
from typing import Tuple
from dataclasses import dataclass

from .base_offscreen_camera import (
    BaseOffscreenCamera,
    _BaseOffCameraMixin
)
from .....vis3d_utils.coordinates import Pose

@dataclass
class OffscreenJunctionFrontCamera(_BaseOffCameraMixin, BaseOffscreenCamera):
    """路口摄像头, 拍摄车头, 也就是正对着 in road
    """
    def init_pos(
        self, pose: Pose, height: float = 10,
        camera_pitch_deg: float = None, camera_setback: float = 0,
        camera_lateral: float = 0,
        *args, **kwargs
    ) -> None:
        """初始化进口道摄像头的空间位置与俯仰朝向。

        相机坐标系说明：
          - element_position（pose）= 停止线中心点 / 上游端点的 SUMO 坐标 (x, y)
          - heading += 270：将 SUMO 行驶方向旋转 270°，使相机正对来车方向
            （SUMO 北=0°顺时针，Panda3D 北=0°逆时针，+270° 完成坐标系适配并翻转朝向）
          - 三向偏移互相正交：setback（纵向后退）、lateral（横向平移）、height（垂直高度）

        Args:
            pose: 元素位姿，由 element_position + element_heading 构造，
                  携带停止线 / 上游端点的 SUMO 坐标与进口道行驶方向。
            height: 相机安装高度（米），对应 SENSOR_CFG 中的 tls_camera_height。
            camera_pitch_deg: 俯仰角（°，向下为正，有效范围 1–89°）。
                None（默认）→ 原始行为：lookAt 目标取 position 前方 0.5m、高度 height//3，
                             等效俯仰角约 44°（height=15m 时）。
                指定值 → 相机以该角度正视地面（z=0），
                         lookAt 水平距离 = height / tan(pitch_deg)。
                建议范围：停止线摄像头 40°–60°，上游道路摄像头 20°–35°。
            camera_setback: 相机沿来车方向的纵向后退量（米，默认 0）。
                0 = 相机位于 element_position 正上方；
                正值 = 向来车方向后退（远离路口），扩大近端纵向视野（建议 5–15m）。
            camera_lateral: 相机沿 stop_line 平行方向的横向偏移量（米，默认 0）。
                正值 = 向中间分隔带方向（内侧）平移；负值 = 向路肩方向（外侧）平移。
                横向偏移不改变相机朝向，仅平移观测角度，适用于强调内侧车道或调整构图。
                几何：横向轴 = heading 方向的右侧垂线，单位向量 = (sin_h, -cos_h)。
        """
        pos, heading = pose.as_panda3d()
        # +270°：SUMO heading → Panda3D heading，并翻转为来车朝向
        # （使相机面朝进口道，即从路口侧看向来车方向）
        heading += 270

        cos_h = np.cos(np.radians(heading))
        sin_h = np.sin(np.radians(heading))

        # ── 横向单位向量（stop_line 平行方向）───────────────────────────────────
        # heading 方向的右侧垂线 (sin_h, -cos_h)，指向中间分隔带（内侧）。
        # 验证：e_in 朝西 heading=0° → (sin0, -cos0)=(0,-1)=南 → 中间分隔带在南侧 ✓
        #       w_in 朝东 heading=180° → (0,+1)=北 → 中间分隔带在北侧 ✓
        lat_x = sin_h
        lat_y = -cos_h

        # ── 相机安装位置：纵向后退 + 横向平移 ──────────────────────────────────
        # 纵向：沿来车方向后退 camera_setback 米（远离路口）
        # 横向：沿 lat 方向偏移 camera_lateral 米（正值 = 内侧/分隔带方向）
        cam_x = pos[0] - camera_setback * cos_h + camera_lateral * lat_x
        cam_y = pos[1] - camera_setback * sin_h + camera_lateral * lat_y
        self.camera_np.setPos(cam_x, cam_y, height)

        if camera_pitch_deg is None:
            # ── 原始 lookAt 模式（camera_pitch_deg=None）────────────────────────
            # lookAt 目标：在 cam 的横向基础上，再向前 0.5m、高度 height//3。
            # 横向偏移与相机同步，保证相机始终正视来车方向（不产生侧偏视角）。
            self.camera_np.lookAt(
                cam_x + 0.5 * cos_h,
                cam_y + 0.5 * sin_h,
                height // 3
            )
        else:
            # ── 俯仰角模式（camera_pitch_deg 已指定）────────────────────────────
            # lookAt 目标：从相机位置（已含横向偏移）沿 heading 方向看向地面（z=0）。
            # 角度夹紧到 [1°, 89°]，防止 tan 趋近 0 或无穷大。
            pitch_rad = math.radians(max(1.0, min(89.0, camera_pitch_deg)))
            horiz_dist = height / math.tan(pitch_rad)
            self.camera_np.lookAt(
                cam_x + horiz_dist * cos_h,
                cam_y + horiz_dist * sin_h,
                0  # 看向地面（z=0）
            )
    
    def update(self, *args, **kwargs) -> None:
        """这里是一个固定的摄像头, 不需要更新位置
        """
        pass
        
    @property
    def position(self) -> Tuple[float, float, float]:
        return self.camera_np.getPos()


@dataclass
class OffscreenJunctionBackCamera(_BaseOffCameraMixin, BaseOffscreenCamera):
    """路口摄像头, 拍摄车尾, 也就是对着 in road 的出口
    """
    def init_pos(self, pose: Pose, height:float=10, *args, **kwargs) -> None:
        pos, heading = pose.as_panda3d()
        heading += 90

        # Calculate the front position based on the vehicle's heading
        self.camera_np.setPos(
            pos[0] - 20 * np.cos(np.radians(heading)),
            pos[1] - 20 * np.sin(np.radians(heading)),
            height
        )

        self.camera_np.lookAt(
            pos[0] - 10.5 * np.cos(np.radians(heading)),
            pos[1] - 10.5 * np.sin(np.radians(heading)),
            height//3
        )
    
    def update(self, *args, **kwargs) -> None:
        """这里是一个固定的摄像头, 不需要更新位置
        """
        pass
        
    @property
    def position(self) -> Tuple[float, float, float]:
        return self.camera_np.getPos()