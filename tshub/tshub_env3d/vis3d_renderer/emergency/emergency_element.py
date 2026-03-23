'''
Author: yufei Ji
Date: 2026-03-22 22:15:44
LastEditTime: 2026-03-23 00:04:07
Description: this script is used to 
FilePath: /VLMTraffic/TransSimHub/tshub/tshub_env3d/vis3d_renderer/emergency/emergency_element.py
'''
import os
from loguru import logger

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
            self.node_path = self.showbase_instance.loader.loadModel(self.model_path)
            # 基础初始化设置
            self.node_path.setPos(self.x, self.y, 0.2)
            
            # --- 根据事件类别针对性地控制缩放系数 ---
            # 普通车辆模型经过转换后通常只需小幅度放大或保持原比例（例如1.0到2.5左右）
            # 不应该统一使用 10.0 这种导致遮挡相邻车道的极端比例
            if self.event_type == 'crash':
                self.node_path.setScale(0.9, 0.9, 0.9) 
            elif self.event_type == 'tree_branch':
                self.node_path.setScale(4, 4, 4)
            else:
                self.node_path.setScale(1.5, 1.5, 1.5)
                
            # SUMO中的绝对方向转为Panda3D的偏航角：
            # SUMO：正北为0，正东为90
            # Panda3D：正北为0，逆时针为正 (setH正值向左偏)，所以为 -heading
            panda_heading = -self.heading
            
            # 设置摆放方向
            if self.event_type == 'tree_branch':
                # 注意：如果树枝原本的3D模型在 Blender里的朝向是垂直的，也可以再加上特定补偿值如 panda_heading + 90
                # 如果使用 -45 度时树木正好平行但压在车道线上，说明模型的原点(Origin)不在中心
                self.node_path.setHpr(panda_heading - 45, 0, 0)
                # 可以通过相对自身坐标系进行平移，把它从车道线移回车道中心 (数值如 1.5 或 -1.5 根据实际偏移方向微调)
                self.node_path.setPos(self.node_path, -1.5, 0, 0)
            else:
                self.node_path.setHpr(panda_heading, 0, 0)
            
            # 为渲染树增加一个独立的 emergency 归属节点（类似于 vehicles、signals）
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
