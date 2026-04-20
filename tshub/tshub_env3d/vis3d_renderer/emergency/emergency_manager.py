'''
Author: yufei Ji
Date: 2026-03-22 22:15:57
LastEditTime: 2026-04-20 17:24:04
Description: this script is used to 
FilePath: /VLMTraffic/TransSimHub/tshub/tshub_env3d/vis3d_renderer/emergency/emergency_manager.py
'''
from .emergency_element import Emergency3DElement, ClosureZone3DElement

class EmergencyManager3D:
    """管理和同步 3D 渲染画面中的所有紧急事件要素"""
    def __init__(self, showbase_instance, root_np, show_closure_zone: bool = False):
        self.showbase_instance = showbase_instance
        self.root_np = root_np
        self.show_closure_zone = show_closure_zone  # 是否在路障前后两端之间渲染半透明封闭区域
        self.rendered_events = {} # dict: { event_id : Emergency3DElement }
        
    def update(self, active_events):
        """每帧调用，根据当前逻辑层的活跃事件，动态进行 3D 模型渲染节点的生命周期管理"""
        active_event_ids = [e['id'] for e in active_events]
        
        # 1. 移除已经不在活跃期内的事件
        for eid in list(self.rendered_events.keys()):
            if eid not in active_event_ids:
                elements = self.rendered_events[eid]
                if isinstance(elements, list):
                    for el in elements:
                        el.remove_node()
                else:
                    elements.remove_node()
                del self.rendered_events[eid]
                
        # 2. 挂载新生事件
        import math
        for event in active_events:
            event_id = event['id']
            if event_id not in self.rendered_events:
                # 检查是否存在基于类型名编码的长度特征 (e.g. barrier_A_5.0)
                e_type = event['type']
                length = 1.0
                
                parts = e_type.split('_')
                if e_type.startswith('barrier_') and len(parts) >= 3:
                    try:
                        e_type = f"{parts[0]}_{parts[1]}"
                        length = float(parts[-1])
                    except ValueError:
                        pass
                
                if length > 1.0 and ('barrier' in e_type):
                    # 识别为连续多模型的路障区间，在当前锚点（事件中心）前后偏移半个 length 的距离
                    half_len = length / 2.0
                    # SUMO中 heading (0=正北, 90=正东): math.sin得 X偏移，math.cos得 Y偏移
                    heading_rad = math.radians(event.get('heading', 0.0))
                    
                    dx = half_len * math.sin(heading_rad)
                    dy = half_len * math.cos(heading_rad)

                    # 后方偏移点 (与行进方向反向)
                    b_x = event['x'] - dx
                    b_y = event['y'] - dy
                    el_back = Emergency3DElement(
                        self.showbase_instance, self.root_np,
                        f"{event_id}_back", e_type, b_x, b_y, event.get('heading', 0.0), event['model_path']
                    )
                    
                    # 前方偏移点 (顺着行进方向)
                    f_x = event['x'] + dx
                    f_y = event['y'] + dy
                    el_front = Emergency3DElement(
                        self.showbase_instance, self.root_np,
                        f"{event_id}_front", e_type, f_x, f_y, event.get('heading', 0.0), event['model_path']
                    )
                    
                    elements = [el_back, el_front]
                    if self.show_closure_zone:
                        zone = ClosureZone3DElement(
                            self.showbase_instance, self.root_np,
                            f"{event_id}_zone",
                            event['x'], event['y'],
                            event.get('heading', 0.0),
                            length=length,
                        )
                        elements.append(zone)
                    self.rendered_events[event_id] = elements
                else:
                    new_element = Emergency3DElement(
                        self.showbase_instance,
                        self.root_np,
                        event_id,
                        e_type,
                        event['x'],
                        event['y'],
                        event.get('heading', 0.0),
                        event['model_path']
                    )
                    self.rendered_events[event_id] = new_element

    def clear(self):
        """重置清除一切渲染节点"""
        for element in self.rendered_events.values():
            if isinstance(element, list):
                for el in element:
                    el.remove_node()
            else:
                element.remove_node()
        self.rendered_events.clear()
