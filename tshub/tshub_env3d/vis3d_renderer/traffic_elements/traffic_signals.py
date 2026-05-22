'''
@Author: WANG Maonan
@Date: 2024-07-07 10:01:04
@Description: 
LastEditTime: 2025-03-25 16:01:17
'''
from typing import Tuple

from .base_element import BaseElement
from ..sensors.rgb_sensor import RGBSensor
from ...vis3d_utils.masks import CamMask

class TLS3DElement(BaseElement):
    def __init__(
            self,
            fig_width: float, fig_height: float,
            fig_resolution: float,
            element_id: str,
            element_position: Tuple[float, float],
            element_heading: float = None,
            element_length: float = None,
            root_np=None,
            showbase_instance=None,
            tls_camera_height: int = 10,
            camera_pitch_deg: float = None,
            camera_setback: float = 0,
            camera_lateral: float = 0,
        ) -> None:
        """模拟路口摄像头

        Args:
            fig_width (float): 传感器输出的图片的宽度
            fig_height (float): 传感器输出的图片的高度
            element_id (str): 信号灯 ID
            element_position (Tuple[float, float]): 信号灯的位置
            element_heading (float, optional): 路口摄像头朝向. Defaults to None.
            element_length (float, optional): 长度, 这里没有用到. Defaults to None.
            root_np: panda3d root path. Defaults to None.
            showbase_instance: panda3d showbase instance. Defaults to None.
            tls_camera_height (int): 相机安装高度（米）. Defaults to 10.
            camera_pitch_deg (float): 俯仰角（°，1–89°）。None → 原始 lookAt 方式. Defaults to None.
            camera_setback (float): 沿来车方向纵向后退距离（米）. Defaults to 0.
            camera_lateral (float): 沿 stop_line 平行方向横向偏移（米）。
                正值 = 向中间分隔带（内侧）平移，负值 = 向路肩（外侧）平移. Defaults to 0.
        """
        super().__init__(
            fig_width, fig_height, fig_resolution,
            element_id, element_position, element_heading, element_length, root_np, showbase_instance
        )
        self.tls_camera_height = tls_camera_height  # 相机安装高度（米），透传给 RGBSensor → init_pos
        # 俯仰角（°，1–89° 有效，None 使用原始 height//3 lookAt）：
        # 控制镜头向地面倾斜程度，值越大越陡，None 保持历史行为
        self.camera_pitch_deg = camera_pitch_deg
        # 沿来车方向的后退偏移（米，默认 0 即位于 element_position 正上方）：
        # 正值使相机后退以扩大近端视野，建议停止线摄像头设 5–15m
        self.camera_setback = camera_setback
        # 沿 stop_line 平行方向的横向偏移（米，默认 0）：
        # 正值 = 向中间分隔带方向（内侧），负值 = 向路肩方向（外侧）
        self.camera_lateral = camera_lateral
    
    def create_node(self) -> None:
        pass

    def update_node(self) -> None:
        pass

    def begin_rendering_node(self) -> None:
        pass

    def attach_sensor_to_element(self, sensor_type: str) -> None:
        sensor_configs = {
            # 拍摄车头
            'junction_front_all': {
                'camera_mask': (CamMask.VehMask | CamMask.GroundMask | CamMask.MapMask | CamMask.SkyBoxMask),
                'camera_type': 'Off_Junction_Front_Camera'
            },
            'junction_front_vehicle': {
                'camera_mask': CamMask.VehMask,
                'camera_type': 'Off_Junction_Front_Camera'
            },
            # 拍摄车尾
            'junction_back_all': {
                'camera_mask': (CamMask.VehMask | CamMask.GroundMask | CamMask.MapMask | CamMask.SkyBoxMask),
                'camera_type': 'Off_Junction_Back_Camera'
            },
            'junction_back_vehicle': {
                'camera_mask': CamMask.VehMask,
                'camera_type': 'Off_Junction_Back_Camera'
            },
        }

        config = sensor_configs.get(sensor_type)
        if config is None:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

        _camera_name = BaseElement._gen_sensor_name(
            base_name=sensor_type, # sensor 类型
            vehicle_id=self.element_id # sensor 放在哪一个 element 上面
        )

        veh_rgb_sensor = RGBSensor(
            camera_name=_camera_name,
            camera_mask=config['camera_mask'],
            showbase_instance=self.showbase_instance,
            root_np=self.root_np,
            init_element_pose=self.get_element_pose_from_center(),
            element_dimensions=(self.length, self.width, self.height),
            fig_width=self.fig_width,
            fig_height=self.fig_height,
            fig_resolution=self.fig_resolution,
            camera_type=config['camera_type'],
            height=self.tls_camera_height,
            camera_pitch_deg=self.camera_pitch_deg,
            camera_setback=self.camera_setback,
            camera_lateral=self.camera_lateral,
        )
        self.sensors[sensor_type] = veh_rgb_sensor
        
    def update_sensor(self) -> None:
        """更新 sensor 的位置, 这里信号灯的摄像机是不需要移动的
        """
        pass
    
    def get_sensor(self):
        sensor_data = {}
        for _sensor_id, _sensor in self.sensors.items():
            ego_rgb = _sensor()
            sensor_data[_sensor_id] = ego_rgb
        return sensor_data