'''
@Author: WANG Maonan
@Date: 2024-07-07 23:30:33
@Description: TSHub 环境的 3D 版本, 整体的逻辑为:
- TshubEnvironment （逻辑层）与 SUMO 进行交互, 获得 SUMO 的数据 (这部分利用 TshubEnvironment)，处理车辆运动、红绿灯逻辑、碰撞检测等。
- TSHubRenderer （视觉层）对 SUMO 的环境进行渲染 (这部分利用 TSHubRenderer)
- TShubSensor 获得渲染的场景的数据, 作为新的 state 进行输出
LastEditTime: 2026-03-06 15:49:03
'''
from loguru import logger
from typing import Any, Dict, List

from .base_env3d import BaseSumoEnvironment3D

from ..tshub_env.tshub_env import TshubEnvironment # tshub 与 sumo 交互
from .vis3d_utils.core_math import calculate_center_point
from .vis3d_renderer.tshub_render import TSHubRenderer # tshub3D render

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
        ) -> None:

        self.debuger_print_node = debuger_print_node
        self.debuger_spin_camera = debuger_spin_camera
        self.should_count_vehicles = should_count_vehicles
        self.is_render = is_render
        self.is_every_frame = is_every_frame

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
        else:
            self.tshub_render = None
            logger.info("SIM: 3D Rendering is DISABLED. Only Physics Simulation will run.")
        
    def reset(self):
        state_infos = self.tshub_env.reset() # 重置 sumo 环境
        logger.info(f'SIM: 完成 TSHub 初始化, 得到地图和信号灯信息.')
        
        if self.is_render:
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
                            'heading': [0.0, 1.0, 0.0],
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
            sensor_data = self.tshub_render.step(states, should_count_vehicles=self.should_count_vehicles)
            # --- 新增功能：计算BEV视角下各个进口道的车道车辆数 ---
            sensor_data['bev_lane_vehicle_counts'] = self._calculate_bev_lane_vehicle_counts(states)
            
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
