from .emergency_element import Emergency3DElement

class EmergencyManager3D:
    """管理和同步 3D 渲染画面中的所有紧急事件要素"""
    def __init__(self, showbase_instance, root_np):
        self.showbase_instance = showbase_instance
        self.root_np = root_np
        self.rendered_events = {} # dict: { event_id : Emergency3DElement }
        
    def update(self, active_events):
        """每帧调用，根据当前逻辑层的活跃事件，动态进行 3D 模型渲染节点的生命周期管理"""
        active_event_ids = [e['id'] for e in active_events]
        
        # 1. 移除已经不在活跃期内的事件
        for eid in list(self.rendered_events.keys()):
            if eid not in active_event_ids:
                element = self.rendered_events[eid]
                element.remove_node()
                del self.rendered_events[eid]
                
        # 2. 挂载新生事件
        for event in active_events:
            if event['id'] not in self.rendered_events:
                new_element = Emergency3DElement(
                    self.showbase_instance,
                    self.root_np,
                    event['id'],
                    event['type'],
                    event['x'],
                    event['y'],
                    event.get('heading', 0.0),
                    event['model_path']
                )
                self.rendered_events[event['id']] = new_element

    def clear(self):
        """重置清除一切渲染节点"""
        for element in self.rendered_events.values():
            element.remove_node()
        self.rendered_events.clear()
