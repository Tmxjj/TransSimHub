'''
Author: yufei Ji
Date: 2026-03-22 22:15:44
LastEditTime: 2026-04-24 23:08:19
Description: this script is used to 紧急事件3D元素
FilePath: /VLMTraffic/TransSimHub/tshub/tshub_env3d/vis3d_renderer/emergency/emergency_element.py
'''
import os
import hashlib
import random as _random_module
from loguru import logger
from panda3d.core import NodePath

class Emergency3DElement:
    def __init__(self, showbase_instance, root_np, event_id, event_type, x, y, heading, model_path):
        self.showbase_instance = showbase_instance
        self.root_np = root_np
        self.event_id = event_id
        self.event_type = event_type
        self.x = x
        self.y = y
        self.heading = heading
        self.model_path = model_path
        self.node_path = None
        
        self.create_node()
        
    def create_node(self):
        """加载紧急事件的三维模型并放置在指定位置"""
        try:
            # 创建一个空的包裹节点(wrapper)作为真实的物理锚点，便于控制整个系统的位置、旋转
            self.node_path = NodePath(f"emergency_wrapper_{self.event_id}")
            
            # 加载实际的源模型并作为它的子节点
            model_np = self.showbase_instance.loader.loadModel(self.model_path)
            model_np.reparentTo(self.node_path)
            
            # 提升模型整体亮度（通过 RGBA 颜色缩放乘数提高亮度，解决渲染过暗的问题）
            model_np.setColorScale(1.5, 1.5, 1.5, 1.0)
            
            # --- 根据事件类别针对性地控制缩放系数 ---
            if self.event_type.startswith(('tree_branch_1lane','tree_branch_3lane')):
                model_np.setScale(1.4, 1.4, 1.4)
            elif self.event_type.startswith(('barrier_A', 'barrier_B', 'barrier_C',
                                     'barrier_D', 'barrier_E')):
                model_np.setScale(1.0, 1.0, 1.0)
            elif self.event_type.startswith('pedestrian_lying'):
                model_np.setScale(1.0, 1.0, 1.0)
            elif self.event_type.startswith('pedestrian_crossing'):
                model_np.setScale(1.0, 1.0, 1.0)
            else:
                model_np.setScale(1.0, 1.0, 1.0)
                
            # 计算缩放后的实际三维包围盒并矫正非居中的锚点问题
            bounds = model_np.getTightBounds()
            if bounds:
                min_b, max_b = bounds
                cx = (min_b[0] + max_b[0]) / 2.0
                cy = (min_b[1] + max_b[1]) / 2.0
                cz = (min_b[2] + max_b[2]) / 2.0
                
                # 反向偏移，把源模型的几何正中心对齐到 wrapper(node_path)的原点坐标
                model_np.setPos(-cx, -cy, -cz)
                
            # 基础初始化设置，给 node_path 设置真实大地坐标和合适的高度
            self.node_path.setPos(self.x, self.y, 0.5)
            
            # SUMO中的绝对方向转为Panda3D的偏航角
            panda_heading = -self.heading

            # crash 车辆：基于 event_id 哈希生成确定性随机偏转角，模拟碰撞后散布姿态
            # 使用 MD5 哈希保证跨仿真/跨进程可复现（不受 PYTHONHASHSEED 影响）
            crash_offset = 0.0
            if self.event_type in ('crash_vehicle_a', 'crash_vehicle_b'):
                _seed = int(hashlib.md5(self.event_id.encode()).hexdigest()[:8], 16)
                crash_offset = _random_module.Random(_seed).uniform(-25.0, 25.0)

            # 设置摆放方向 (统一对 wrapper 层进行旋转和调整)
            if self.event_type in ('pedestrian_lying', 'pedestrian_crossing'):
                # 行人模型面朝行车方向侧面，旋转 90°
                self.node_path.setHpr(panda_heading + 90, 0, 0)
            else:
                self.node_path.setHpr(panda_heading + crash_offset, 0, 0)
            
            # 为渲染树增加一个独立的 emergency 归属节点
            parent_node = self.root_np.find("**/emergency")
            if parent_node.isEmpty():
                parent_node = self.root_np.attachNewNode("emergency")
                
            self.node_path.reparentTo(parent_node)
            logger.info(f"🚧 [+3D Node] 紧急事件渲染成功: {self.event_id} ({self.event_type}) 坐标: ({self.x}, {self.y})")
        except Exception as e:
            logger.error(f"SIM: 加载事件模型失败 {self.model_path}: {e}")
            self.node_path = None
            
    def remove_node(self):
        """移除渲染节点"""
        if self.node_path:
            self.node_path.removeNode()
            self.node_path = None
            logger.info(f"🚧 [-3D Node] 紧急事件撤销渲染: {self.event_id}")


class ClosureZone3DElement:
    """在两个路障之间渲染半透明矩形封闭区域"""
    def __init__(self, showbase_instance, root_np, zone_id, x, y, heading, length, width=3.5):
        self.showbase_instance = showbase_instance
        self.root_np = root_np
        self.zone_id = zone_id
        self.x = x
        self.y = y
        self.heading = heading
        self.length = length
        self.width = width
        self.node_path = None

        self.create_node()

    def create_node(self):
        from panda3d.core import CardMaker, TransparencyAttrib, LColor
        try:
            cm = CardMaker(f"closure_zone_{self.zone_id}")
            half_l = self.length / 2.0
            half_w = self.width / 2.0
            # XZ 平面卡片：X 轴为宽度方向，Z 轴为长度方向
            cm.setFrame(-half_w, half_w, -half_l, half_l)
            cm.setColor(LColor(1.0, 0.55, 0.0, 0.55))

            self.node_path = NodePath(f"closure_zone_wrapper_{self.zone_id}")
            card_np = self.node_path.attachNewNode(cm.generate())
            card_np.setTwoSided(True)
            card_np.setShaderOff(1)  # 脱离 simplepbr 继承的 PBR shader，确保透明度生效
            card_np.setTransparency(TransparencyAttrib.MAlpha)

            # 绕 X 轴旋转 -90°，将卡片从竖直变为水平（XZ→XY）
            # 再用 panda_heading 对齐行车方向
            panda_heading = -self.heading
            self.node_path.setHpr(panda_heading, 0, 0)
            card_np.setHpr(0, -90, 0)

            self.node_path.setPos(self.x, self.y, 0.15)

            parent_node = self.root_np.find("**/emergency")
            if parent_node.isEmpty():
                parent_node = self.root_np.attachNewNode("emergency")
            self.node_path.reparentTo(parent_node)
            logger.info(f"🟧 [+3D Zone] 路障封闭区域渲染: {self.zone_id} 中心({self.x:.1f},{self.y:.1f}) 长={self.length}m 宽={self.width}m")
        except Exception as e:
            logger.error(f"SIM: 路障封闭区域渲染失败 {self.zone_id}: {e}")
            self.node_path = None

    def remove_node(self):
        if self.node_path:
            self.node_path.removeNode()
            self.node_path = None
            logger.info(f"🟧 [-3D Zone] 路障封闭区域撤销: {self.zone_id}")
