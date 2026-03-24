'''
Author: yufei Ji
Date: 2026-03-22 22:15:44
LastEditTime: 2026-03-24 20:02:54
Description: this script is used to 紧急事件3D元素
FilePath: /VLMTraffic/TransSimHub/tshub/tshub_env3d/vis3d_renderer/emergency/emergency_element.py
'''
import os
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
            if self.event_type == 'crash':
                model_np.setScale(0.8, 0.9, 0.9) 
            elif self.event_type == 'tree_branch':
                model_np.setScale(3.5, 3, 4)
            else:
                model_np.setScale(1.5, 1.5, 1.5)
                
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
            
            # 设置摆放方向 (统一对 wrapper 层进行旋转和调整)
            if self.event_type == 'tree_branch':
                self.node_path.setHpr(panda_heading - 45, 0, 0)
            else:
                self.node_path.setHpr(panda_heading, 0, 0)
            
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
