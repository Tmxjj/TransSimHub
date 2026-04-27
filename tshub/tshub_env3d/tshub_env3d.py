'''
@Author: WANG Maonan
@Date: 2024-07-07 23:30:33
@Description: TSHub 环境的 3D 版本, 整体的逻辑为:
- TshubEnvironment （逻辑层）与 SUMO 进行交互, 获得 SUMO 的数据 (这部分利用 TshubEnvironment)，处理车辆运动、红绿灯逻辑、碰撞检测等。
- TSHubRenderer （视觉层）对 SUMO 的环境进行渲染 (这部分利用 TSHubRenderer)
- TShubSensor 获得渲染的场景的数据, 作为新的 state 进行输出
LastEditTime: 2026-04-19 21:48:36
'''
import os
import sys
from loguru import logger
from typing import Any, Dict, List

from .base_env3d import BaseSumoEnvironment3D
from ..tshub_env.tshub_env import TshubEnvironment # tshub 与 sumo 交互
from .vis3d_utils.core_math import calculate_center_point
from .vis3d_renderer.tshub_render import TSHubRenderer # tshub3D render

import xml.etree.ElementTree as ET
import os

current_dir = os.path.dirname(os.path.abspath(__file__))


class EventManager:
    """
    解析路由文件中的事件参数（<param> 标签），按时间窗口向 EmergencyManager3D 提供活跃事件列表。
    仅负责"读取"与"过滤"，不负责生成路由文件（生成由 scripts/event_scene_generation/ 完成）。
    """
    def __init__(self):
        self.active_events = []

    def load_events_from_xml(self, route_xml_path: str) -> list:
        """解析 .rou.xml，提取含 pos_x/pos_y 参数的 <trip> 作为可渲染事件"""
        self.active_events = []
        if not os.path.exists(route_xml_path):
            logger.warning(f"SIM: Event route file not found: {route_xml_path}")
            return self.active_events

        try:
            tree = ET.parse(route_xml_path)
            root = tree.getroot()
            for trip in root.findall("trip"):
                params = {p.get("key"): p.get("value") for p in trip.findall("param")}
                if "pos_x" not in params or "pos_y" not in params:
                    continue
                stop = trip.find("stop")
                duration = float(stop.get("duration", 0)) if stop is not None else 0.0
                start_time = float(trip.get("depart", 0))
                self.active_events.append({
                    'id':         trip.get("id"),
                    'type':       params.get("event_type", "unknown"),
                    'x':          float(params["pos_x"]),
                    'y':          float(params["pos_y"]),
                    'heading':    float(params.get("heading", 0.0)),
                    'start_time': start_time,
                    'end_time':   start_time + duration,
                    'model_path': params.get("model_path", ""),
                })
        except Exception as e:
            logger.warning(f"SIM: Failed to parse event route file {route_xml_path}: {e}")

        logger.info(f"SIM: Loaded {len(self.active_events)} events from {route_xml_path}")
        return self.active_events

    def get_active_events(self, current_time: float) -> list:
        """返回当前仿真时刻处于激活时间窗内的事件"""
        return [e for e in self.active_events
                if e['start_time'] <= current_time <= e['end_time']]

# 辅助函数，从sumocfg中去netfile的路径
def get_sumo_net_file(config_file):
    try:
        # 1. 解析 XML 文件
        tree = ET.parse(config_file)
        root = tree.getroot()

        # 2. 寻找 input 标签下的 net-file 标签
        # .sumocfg 的结构通常是 configuration -> input -> net-file
        net_file_node = root.find(".//net-file")

        if net_file_node is not None:
            # 3. 获取 value 属性
            net_file_path = net_file_node.get("value")
            return net_file_path
        else:
            print(f"❌ 错误: 在 {config_file} 中未找到 <net-file> 标签")
            return None

    except ET.ParseError:
        print(f"❌ 错误: 无法解析文件 {config_file}，请检查是否为标准 XML 格式")
        return None
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {config_file}")
        return None


def get_absolute_net_path(sumocfg_path):
    net_val = get_sumo_net_file(sumocfg_path)
    if net_val:
        # 获取 .sumocfg 所在的文件夹目录
        base_dir = os.path.dirname(os.path.abspath(sumocfg_path))
        # 拼接成绝对路径并规范化（处理 ./ 或 ../）
        abs_net_path = os.path.normpath(os.path.join(base_dir, net_val))
        return abs_net_path
    return None

class Tshub3DEnvironment(BaseSumoEnvironment3D):
    def __init__(
            # TshubEnvironment 的参数 (与 SUMO 交互)
            self, sumo_cfg: str, 
            scenario_glb_dir: str, # 场景 3D 模型存储的位置
            is_map_builder_initialized: bool = False, 
            is_vehicle_builder_initialized: bool = True, 
            is_aircraft_builder_initialized: bool = True, 
            is_traffic_light_builder_initialized: bool = True, 
            is_person_builder_initialized: bool = True, 
            poly_file: str = None, 
            osm_file: str = None, 
            radio_map_files: Dict[str, str] = None, 
            tls_ids: List[str] = None, 
            aircraft_inits: Dict[str, Any] = None, 
            vehicle_action_type: str = 'lane', 
            hightlight: bool = False, 
            tls_action_type: str = 'next_or_not', 
            delta_time: int = 5, 
            net_file: str = None, route_file: str = None, 
            trip_info: str = None, statistic_output: str = None, 
            summary: str = None, queue_output: str = None, 
            tls_state_add: List = None, 
            use_gui: bool = False, is_libsumo: bool = False, 
            begin_time=0, num_seconds=20000, 
            max_depart_delay=100000, time_to_teleport=-1, sumo_seed: str = 'random', 
            tripinfo_output_unfinished: bool = True, 
            collision_action: str = None, # 车辆发生碰撞之后做的事情
            remote_port: int = None, 
            num_clients: int = 1,
            # TSHubRenderer 的参数
            preset:str = '480P', 
            resolution:float = 0.5,
            vehicle_model:str='low', # 车辆加载模型, low 或是 high
            render_mode: str = "onscreen",
            should_count_vehicles: bool = False, # 是否返回的时候获得车辆信息, 将车辆信息保存为 JSON 进行额外的渲染
            debuger_print_node:bool = False, # 是否在 reset 的时候打印 node path
            debuger_spin_camera:bool = False, # 是否显示 spin camera
            sensor_config: Dict[str, List[str]] = None,
            is_render: bool = True, # 是否渲染
            is_every_frame: bool = False, # 是否每一帧都渲染
            show_closure_zone: bool = True, # 是否在路障占道区间渲染半透明封闭区域
        ) -> None:

        self.debuger_print_node = debuger_print_node
        self.debuger_spin_camera = debuger_spin_camera
        self.should_count_vehicles = should_count_vehicles
        self.is_render = is_render
        self.is_every_frame = is_every_frame

        # --- 1. 读取离线生成的紧急事件文件 ---
        self.event_logic_manager = None
        if EventManager is not None and sumo_cfg:
            # 找到当前 sumocfg 文件内部加载的 rou.xml，读取经过合并的紧急事件
            sumocfg_dir = os.path.dirname(os.path.abspath(sumo_cfg))
            route_val = None
            try:
                tree = ET.parse(sumo_cfg)
                root = tree.getroot()
                route_file_node = root.find(".//route-files")
                if route_file_node is not None:
                    route_val = route_file_node.get("value").split(',')[0]
            except Exception as e:
                logger.warning(f"SIM: Failed to parse sumocfg for route-files: {e}")

            if route_val:
                route_xml_path = os.path.normpath(os.path.join(sumocfg_dir, route_val))
                if os.path.exists(route_xml_path):
                    self.event_logic_manager = EventManager()
                    self.event_logic_manager.load_events_from_xml(route_xml_path)

                    # 检查是否成功加载了事件，将相对路径转换为绝对路径
                    if len(self.event_logic_manager.active_events) > 0:
                        base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
                        for e in self.event_logic_manager.active_events:
                            e['model_path'] = os.path.normpath(os.path.join(base_dir, e['model_path']))
                        logger.info(f"SIM: Successfully loaded {len(self.event_logic_manager.active_events)} offline events from: {route_xml_path}")
                else:
                    logger.info(f"SIM: Route file not found at {route_xml_path}")

        # 初始化 tshub 环境与 sumo 交互
        self.tshub_env = TshubEnvironment(
            sumo_cfg, 
            is_map_builder_initialized, 
            is_vehicle_builder_initialized, 
            is_aircraft_builder_initialized, 
            is_traffic_light_builder_initialized, 
            is_person_builder_initialized, 
            poly_file, osm_file, radio_map_files, tls_ids, aircraft_inits, 
            vehicle_action_type, hightlight, tls_action_type, delta_time, 
            net_file, route_file, trip_info, statistic_output, summary, queue_output, 
            tls_state_add, use_gui, is_libsumo, begin_time, num_seconds, max_depart_delay, 
            time_to_teleport, sumo_seed, tripinfo_output_unfinished, collision_action, 
            remote_port, num_clients
        )
        # 记录虚拟 aircraft 高度配置（如未配置则默认 80m）
        self.aircraft_bev_height = 80.0
        try:
            self.aircraft_bev_height = sensor_config.get('aircraft', {}) \
                .get(f'aircraft_{tls_ids[0]}', {}) \
                .get('height', 80.0)
        except Exception:
            pass
        # 记录 BEV 相机朝向配置（如未配置则默认 [0,1,0] 即正北朝上）
        # 通过 SENSOR_CFG 中 aircraft.camera_heading 可按场景自定义旋转角度：
        #   [0,1,0]  → heading=0°  → 正北朝上（默认）
        #   [-1,1,0] → heading=45° → 顺时针旋转 45°（适用于道路走向 NE/NW/SE/SW 的场景）
        # 注意：sensor_config 经 tsc_env.py 转换后结构为 {'aircraft': {'aircraft_J2': {...}}}，
        #       因此必须用 .get(f'aircraft_{tls_ids[0]}', {}) 定位到具体路口的配置项。
        self.aircraft_camera_heading = [0.0, 1.0, 0.0]
        try:
            heading_cfg = sensor_config.get('aircraft', {}) \
                .get(f'aircraft_{tls_ids[0]}', {}) \
                .get('camera_heading', None)
            if heading_cfg is not None:
                self.aircraft_camera_heading = [float(v) for v in heading_cfg]
        except Exception:
            pass

        # 初始化渲染器, 将场景渲染为 3D
        if self.is_render:
            self.tshub_render = TSHubRenderer(
                simid=f"tshub-{self.tshub_env.CONNECTION_LABEL}", # 场景的 ID
                scenario_glb_dir=scenario_glb_dir,
                sensor_config=sensor_config,
                preset=preset,
                resolution=resolution,
                render_mode=render_mode,
                vehicle_model=vehicle_model,
            )
            
            # --- 2. 初始化视觉层上的 3D 紧急事件管理器 ---
            from .vis3d_renderer.emergency.emergency_manager import EmergencyManager3D
            self.emergency_renderer = EmergencyManager3D(
                self.tshub_render._showbase_instance,
                self.tshub_render._root_np,
                show_closure_zone=show_closure_zone,
            )
        else:
            self.tshub_render = None
            self.emergency_renderer = None
            logger.info("SIM: 3D Rendering is DISABLED. Only Physics Simulation will run.")
        
    def reset(self):
        state_infos = self.tshub_env.reset() # 重置 sumo 环境
        logger.info(f'SIM: 完成 TSHub 初始化, 得到地图和信号灯信息.')
        
        if self.is_render:
            if getattr(self, 'emergency_renderer', None) is not None:
                self.emergency_renderer.clear()
            self.tshub_render.reset(state_infos) # 重置 render, 需要将信号灯的信息传入, 辅助进行路口 camera 的初始化

            # 加入一个简单任务, 避免 userExit 出错
            self.tshub_render._showbase_instance.taskMgr.add(
                self.tshub_render.dummyTask, "dummyTask"
            )

            # 重置后打印 node path (查看每次 reset 是否会重置所有 node 和 camera)
            if self.debuger_print_node:
                self.tshub_render.print_node_paths(self.tshub_render._root_np)

            # 场景添加相机, 可以进行可视化
            if self.debuger_spin_camera:
                self.tshub_render._showbase_instance.taskMgr.add(
                    self.tshub_render.test_spin_camera_task, 
                    "SpinCamera"
                )

        return state_infos
    
    def _calculate_bev_lane_vehicle_counts(self, states):
        #NOTE：该函数的目的是计算在 BEV 视角下, 各个进口车道上真实存在的车辆数量, 只针对于杭州、济南、纽约这些规整的数据集
        """计算BEV图片覆盖范围内，各个进口车道上的真实车辆数"""
        bev_lane_vehicle_counts = {}
        try:
            veh_states = states.get('vehicle', {})
            tls_states = states.get('tls', {})
            tls_ids_list = self.tshub_env.tls_ids if self.tshub_env.tls_ids else []
            
            for tls_id in tls_ids_list:
                if tls_id not in tls_states:
                    continue
                tls_info = tls_states[tls_id]
                stop_lines = tls_info.get('in_road_stop_line', {})
                if not stop_lines:
                    continue
                
                # 获取路口中心点
                center_list = []
                for i in range(len(stop_lines)):
                    center_list.append(calculate_center_point(stop_lines[list(stop_lines.keys())[i]]))
                center = calculate_center_point(center_list)
                
                import math
                # --- 计算真实的 BEV 相机地面视野边界范围 ---
                fov_h = 90 # 目前构建相机时使用的水平视野角 (FOV) 为 90 度
                fig_w, fig_h = 800, 600 # 默认长宽
                if self.is_render and self.tshub_render and hasattr(self.tshub_render, 'scene_sync'):
                    fig_w = self.tshub_render.scene_sync.fig_width
                    fig_h = self.tshub_render.scene_sync.fig_height
                aspect_ratio = fig_w / fig_h
                
                # 依据三角函数，计算中心点到视野矩形边缘的横向/纵向物理距离
                coverage_x = self.aircraft_bev_height * math.tan(math.radians(fov_h / 2))
                coverage_y = coverage_x / aspect_ratio
                # ------------------------------------------
                
                # 确定需要统计的目标进口车道
                target_lanes = set()
                in_roads = tls_info.get('in_roads', [])
                roads_lanes = tls_info.get('roads_lanes', {})
                if roads_lanes:
                    for road in in_roads:
                        lanes = roads_lanes.get(road, [])
                        for lane in lanes:
                            target_lanes.add(lane)
                            
                # 初始化车道计数字典
                lane_counts = {lane: 0 for lane in target_lanes}
                
                # 遍历所有车辆并进行边界框筛选计数
                if target_lanes:
                    for veh_id, veh_info in veh_states.items():
                        veh_lane = veh_info.get('lane_id')
                        if veh_lane in target_lanes:
                            veh_pos = veh_info.get('position')
                            if veh_pos and center:
                                dx = abs(veh_pos[0] - center[0])
                                dy = abs(veh_pos[1] - center[1])
                                # 判断车辆是否在当前相机的地面矩形可见范围内
                                if dx <= coverage_x and dy <= coverage_y:
                                    lane_counts[veh_lane] += 1
                                    
                bev_lane_vehicle_counts[f'aircraft_{tls_id}'] = lane_counts
        except Exception as e:
            logger.warning(f"SIM: Failed to calculate BEV lane vehicle counts: {e}")
            
        return bev_lane_vehicle_counts

    def step(self, actions):
        # BUG：states中包含 vehicle, tls, aircraft 三个部分的数据，其中vehicle数据无法保证都在同一个路口的BEV视角下
        # 1. 与 SUMO 进行交互
        states, rewards, infos, dones = self.tshub_env.step(actions)

        # 1.5 注入虚拟 aircraft 用于 BEV 俯视相机
        try:
            tls_ids = self.tshub_env.tls_ids if self.tshub_env.tls_ids else []
            for tls_id in tls_ids:
                if tls_id and 'tls' in states and tls_id in states['tls']:
                    # 获取路口中心点
                    stop_lines = states['tls'][tls_id].get('in_road_stop_line', {})
                    if stop_lines:
                        center_list = []
                        # 先去计算所有 stopline 的中心点, 再计算整体的中心点
                        for i in range(len(stop_lines)):
                            center_list.append(calculate_center_point(stop_lines[list(stop_lines.keys())[i]]) )
                        center = calculate_center_point(center_list)
                        bev_height = self.aircraft_bev_height
                        states.setdefault('aircraft', {})
                        states['aircraft'][f'aircraft_{tls_id}'] = {
                            'position': [center[0], center[1], bev_height],
                            'heading': self.aircraft_camera_heading,
                        }
        except Exception as e:
            logger.warning(f"SIM: Inject aircraft for BEV camera failed: {e}")
            # 失败时不影响主流程
            pass

        # 2. 渲染 3D 的场景

        if self.is_every_frame:
            can_perform_action = True
        else: 
            # 当前仿真时间点可以执行动作时才渲染
            # 支持多路口: 只要有一个路口可以执行动作(且需要渲染), 则进行渲染
            can_perform_action = False
            tls_ids = self.tshub_env.tls_ids if self.tshub_env.tls_ids else []
            for tls_id in tls_ids:
                if tls_id in states['tls'] and states['tls'][tls_id]['can_perform_action']:
                    can_perform_action = True
                    break
        if self.is_render and self.tshub_render and can_perform_action:

            # 先更新事件场景节点，再渲染帧，确保事件模型出现在当帧图像中
            if getattr(self, 'event_logic_manager', None) is not None and getattr(self, 'emergency_renderer', None) is not None:
                current_time = self.tshub_env.sim_step
                active_events = self.event_logic_manager.get_active_events(current_time)
                self.emergency_renderer.update(active_events)

            sensor_data = self.tshub_render.step(states, should_count_vehicles=self.should_count_vehicles)
            # --- 新增功能：计算BEV视角下各个进口道的车道车辆数 ---
            # sensor_data['bev_lane_vehicle_counts'] = self._calculate_bev_lane_vehicle_counts(states)
            
        else:
            # 组装 sensor_data (不包含 image)
            sensor_data = {
                'image': None,
                'veh_elements': None,
                'bev_lane_vehicle_counts': None
            }

        return states, rewards, infos, dones, sensor_data
        
       

    def close(self) -> None:
        self.tshub_env._close_simulation()
        if self.is_render and self.tshub_render:
            self.tshub_render.destroy()
