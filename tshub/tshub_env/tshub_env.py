'''
@Author: WANG Maonan
@Date: 2023-08-23 15:34:52
@Description: 整合 "Veh"（车辆）、"Air"（航空）和 "Traf"（信号灯）的环境
LastEditTime: 2026-03-06 11:15:39
'''
import os
import sys
from loguru import logger
from typing import Dict, List, Any, Literal

from .base_sumo_env import BaseSumoEnvironment
from ..map.map_builder import MapBuilder
from ..aircraft.aircraft_builder import AircraftBuilder
from ..traffic_light.traffic_light_builder import TrafficLightBuilder
from ..vehicle.vehicle_builder import VehicleBuilder
from ..person.person_builder import PersonBuilder
from ..visualization.visualize_map import render_map
from ..visualization.filter_objects import filter_object

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


class TshubEnvironment(BaseSumoEnvironment):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param begin_time: (int) The time step (in seconds) the simulation starts
    :param num_seconds: (int) Number of simulated seconds on SUMO. The time in seconds the simulation must end.
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    :sumo_seed: (int/string) Random seed for sumo. If 'random' it uses a randomly chosen seed.
    :fixed_ts: (bool) If true, it will follow the phase configuration in the route_file and ignore the actions. 不调整信号灯
    """

    def __init__(self, 
                 sumo_cfg: str, 
                 is_map_builder_initialized:bool = False,
                 is_vehicle_builder_initialized:bool = True, 
                 is_aircraft_builder_initialized:bool = True, 
                 is_traffic_light_builder_initialized:bool = True,
                 is_person_builder_initialized:bool = True,
                 poly_file:str = None, osm_file:str = None, radio_map_files:Dict[str, str]=None,
                 tls_ids:List[str] = None, aircraft_inits:Dict[str, Any] = None,
                 vehicle_action_type:str = 'lane', hightlight:bool = False,
                 tls_action_type:str = 'next_or_not', delta_time:int=5,
                 net_file: str = None, route_file: str = None, 
                 trip_info: str = None, statistic_output: str = None, summary: str = None, queue_output: str = None, 
                 tls_state_add: List = None, use_gui: bool = False, is_libsumo: bool = False, 
                 begin_time=0, num_seconds=20000, max_depart_delay=100000, time_to_teleport=-1, 
                 sumo_seed: str = 'random', tripinfo_output_unfinished:bool=True, collision_action:str=None,
                 remote_port: int = None, num_clients: int = 1
        ) -> None:
        
        super().__init__(sumo_cfg, net_file, route_file, 
                         trip_info, statistic_output, summary, queue_output, 
                         tls_state_add, use_gui, is_libsumo, 
                         begin_time, num_seconds, max_depart_delay, time_to_teleport, 
                         sumo_seed, tripinfo_output_unfinished, 
                         collision_action, remote_port, num_clients
                        )

        self.is_map_builder_initialized = is_map_builder_initialized
        self.is_vehicle_builder_initialized = is_vehicle_builder_initialized
        self.is_aircraft_builder_initialized = is_aircraft_builder_initialized
        self.is_traffic_light_builder_initialized = is_traffic_light_builder_initialized
        self.is_person_builder_initialized = is_person_builder_initialized

        # Map Builder Input
        self.poly_file = poly_file
        self.osm_file = osm_file
        self.radio_map_files = radio_map_files

        # Traffic Light Builder Input
        self.tls_ids = tls_ids
        self.tls_action_type = tls_action_type
        self.delta_time = delta_time
        if self.is_traffic_light_builder_initialized is True and not self.tls_ids:
            raise ValueError("Both `map_init` and `tls_ids` need to be set together.")
        if tls_ids is not None:
            if not isinstance(self.tls_ids, list):
                raise TypeError("The 'tls_ids' must be of type 'list'.")

        # Aircraft Builder Input
        self.aircraft_inits = aircraft_inits
        
        # Vehicle Builder Input
        self.vehicle_action_type = vehicle_action_type
        self.hightlight = hightlight

        # For SUMI-GUI render
        self.render_count = 0

    def __init_builder(self) -> None:
        map_builder = (
            MapBuilder(net_file=self._net, poly_file=self.poly_file, osm_file=self.osm_file, radio_map_files=self.radio_map_files)
            if self.is_map_builder_initialized
            else None
        )
        if map_builder is not None:
            self.map_infos = map_builder.get_objects_infos() # Statistic Map Info

        vehicle_builder = (
            VehicleBuilder(sumo=self.sumo, action_type=self.vehicle_action_type, hightlight=self.hightlight)
            if self.is_vehicle_builder_initialized
            else None
        )
        aircraft_builder = (
            AircraftBuilder(sumo=self.sumo, aircraft_inits=self.aircraft_inits)
            if self.is_aircraft_builder_initialized
            else None
        )
        tls_builder = (
            TrafficLightBuilder(sumo=self.sumo, tls_ids=self.tls_ids, action_type=self.tls_action_type, delta_time=self.delta_time)
            if self.is_traffic_light_builder_initialized
            else None
        )
        person_builder = (
            PersonBuilder(sumo=self.sumo)
            if self.is_person_builder_initialized
            else None
        )

        self.scene_objects = {
            'vehicle': vehicle_builder,
            'aircraft': aircraft_builder,
            'tls': tls_builder, # Traffic Light Signal
            'person': person_builder,
        }

    def reset(self) -> Dict[str, Any]:
        """重置环境, 返回初始的 obs
        """
        self._close_simulation() # 关闭仿真
        self._start_simulation() # 开启仿真
        self.__init_builder() # 初始化场景内的 builder
        obs = self.__computer_observation()

        self.obs = obs.copy() # copy obs for render

        return obs
    
    def step(self, actions):
        # apply action
        for _object_type, _object_action in actions.items():
            if self.scene_objects[_object_type] is not None:
                self.scene_objects[_object_type].control_objects(_object_action)
        
        self.sumo.simulationStep()
        logger.info(f'SIM: ==> Simulation Step: {self.sim_step} <==') # 日志中打印当前的仿真时间

        # update env
        obs = self.__computer_observation()
        info = self.__compute_info()
        done = self._computer_done()

        self.obs = obs.copy() # copy obs for render
        reward = self.__computer_reward()
        
        return obs, reward, info, done

    def __computer_observation(self) -> Dict[str, Any]:
        """自定义 obs 的计算
        """
        env_state = {
            _object_type: _object_builder.get_objects_infos()
            for _object_type, _object_builder in self.scene_objects.items()
            if _object_builder is not None
        }
        if self.is_map_builder_initialized:
            env_state.update(self.map_infos) # 地图信息是固定的, 只需要每次额外补充进去即可, 不需要每次计算
        return env_state
    # TODO：完善 reward 计算方式
    def __computer_reward(self) -> float:
        """自定义 reward 的计算 (单步)
        基于 self.obs 中的 TLS 数据计算综合指标:
        Reward = w_speed * 平均速度 - w_queue * 排队指数
        排队指数：
        1、“完全停下的排队长度”：直接使用 tls_obs['J1']['jam_length_vehicle'] 累加即可
        2、“包含缓慢移动的拥堵程度”：建议结合 last_step_mean_speed 和 last_step_occupancy 来综合判断
        """
        total_rewards_dict = {}
        
        # 权重参数
        w_queue = 1.0   # 排队惩罚权重
        w_speed = 0.5   # 速度奖励权重

        # 获取 TLS 观测数据
        tls_obs = self.obs.get('tls', {})
        
        for tls_id, tls_info in tls_obs.items():
            # 提取关键指标: 速度和占用率
            # 注意: last_step_mean_speed 为 -1 表示该车道无车
            mean_speeds = tls_info.get('last_step_mean_speed', [])
            occupancies = tls_info.get('last_step_occupancy', [])
            
            # 1. 计算排队指数 (Queue Score)
            # 逻辑: 如果车道占用率 > 1% 且 平均速度 < 0.1 m/s，视为排队
            current_queue_score = 0
            for speed, occ in zip(mean_speeds, occupancies):
                if occ > 1.0 and speed < 0.1 and speed != -1:
                    # 估算: 占用率每 5% 约等于 1 辆积压车辆
                    current_queue_score += (occ / 5.0)
            
            # 2. 计算平均通行速度 (Average Speed)
            # 只统计有车的车道 (speed >= 0)
            valid_speeds = [s for s in mean_speeds if s >= 0]
            avg_speed = sum(valid_speeds) / len(valid_speeds) if valid_speeds else 0
            
            # 3. 聚合单路口奖励
            # 速度越快越好(正)，排队越少越好(负)
            step_reward = (w_speed * avg_speed) - (w_queue * current_queue_score)
            total_rewards_dict[tls_id] = step_reward

        return total_rewards_dict

    def __compute_info(self):
        """每一步, 返回信息
        """
        return {
            'step_time': self.sim_step, # 返回当前仿真的时间
        }
    # NOTE:默认为rgb，可以尝试sumo_gui
    def render(self, mode:str='rgb',
               focus_id:str=None, focus_type:str=None, focus_distance:float=None, 
               save_folder:str=None
        ) -> None:
        """对场景进行渲染

        Args:
            mode (str, optional): 渲染的模式，包含 rgb 和 sumo_gui. Defaults to rgb.
            focus_id (str, optional): 追踪模式，设置追踪 object 的 ID. Defaults to None. 如果设置为 None，就是全局渲染
            focus_type (str, optional): 追踪 object 的类型，包含 vehicle 和 node. Defaults to None.
            focus_distance (float, optional): 追踪覆盖的范围. Defaults to None.
            save_folder (str, optional): 当 mode='sumo_gui' 的时候，图像保存的文件夹。 
        """
        if not self.is_map_builder_initialized:
            raise ValueError('需要初始化地图信息')
        
        # Step 1. Filter Object (找出符合要求的 object 坐标)
        obs, x_range, y_range = filter_object(
            self.obs, 
            focus_id, focus_type, focus_distance
        )

        # Step 2. Render Image (rgb or sumo-gui)
        if mode == 'rgb':
            if (x_range is None) and (y_range is None):
                fig = None # 如果追踪的物体不在, 则 fig 直接返回 None
            else:
                map_lanes, map_nodes, vehicle_info = obs['lane'], obs['node'], obs['vehicle']
                fig = render_map(
                    focus_id,
                    map_lanes, map_nodes, vehicle_info, 
                    x_range=x_range, y_range=y_range
                ) # 如果在, 则渲染 focus_id 附近的内容
            return fig
        elif mode == 'sumo_gui':
            assert self.use_gui == True, '需要开启 GUI 界面才可以使用 SUMO-GUI 进行渲染。'
            assert save_folder is not None, '需要在 save_folder 设置文件保存的路径。'

            if self.render_count == 0: # 第一次初始化时候需要创建新的 view
                logger.warning(f'使用 SUMO-GUI 截图前确保窗口最大化。')
                import time; time.sleep(5)
                self.traci.gui.removeView('View #0')
                self.traci.gui.addView('RenderView', schemeName="real world")
            
            if (x_range is None) and (y_range is None):
                pass
            elif self.render_count>0:
                self.traci.gui.setBoundary(
                    viewID='RenderView', 
                    xmin=x_range[0], ymin=y_range[0], 
                    xmax=x_range[1], ymax=y_range[1]
                ) # 设置范围
                self.traci.gui.screenshot(
                    viewID='RenderView', 
                    filename=f'{save_folder}/{self.render_count}.png', 
                    width=600, height=600
                )
            self.render_count += 1
            return None
        else:
            raise ValueError(f'mode can only be rgb and sumo_gui, now is {mode}.')

    def _load_state(self, filename: str) -> None:
        """重写父类的 _load_state，在回滚后重新订阅传感器"""
        super()._load_state(filename)
        # 重新订阅 TLS 传感器 (workaround for subscription loss after loadState)
        if self.scene_objects.get('tls') is not None:
             self.scene_objects['tls'].subscribe_detector()
        if self.scene_objects.get('vehicle') is not None:
             self.scene_objects['vehicle'].subscribe_all_vehicles()
        if self.scene_objects.get('person') is not None:
             self.scene_objects['person'].subscribe_all_persons()