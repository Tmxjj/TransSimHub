'''
@Author: WANG Maonan
@Date: 2024-07-03 23:43:42
@Description: 继承 ShowBase, Panda3D 的主界面
LastEditTime: 2026-03-16 11:31:43
'''
from ...utils.get_abs_path import get_abs_path
current_file_path = get_abs_path(__file__)

import simplepbr
from loguru import logger
from threading import Lock
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GraphicsPipeSelection

from panda3d.core import (
    NodePath,
    Shader,
    loadPrcFileData,
)

from .base_render import DEBUG_MODE, BACKEND_LITERALS

class _ShowBaseInstance(ShowBase):
    """Wraps a singleton instance of ShowBase from Panda3D.
    """
    _debug_mode: DEBUG_MODE = DEBUG_MODE.WARNING
    _rendering_backend: BACKEND_LITERALS = "p3headlessgl" # pandagl, p3headlessgl
    _render_mode: str = "onscreen" # onscreen or offscreen

    @classmethod
    def load_config(cls, key, value) -> None:
        """Helper method to load configuration.
        """
        loadPrcFileData("", f"{key} {value}")
        
    def __new__(cls, use_render_pipeline=False):
        # Singleton pattern:  ensure only 1 ShowBase instance
        if "__it__" not in cls.__dict__:
            # ==========================================
            # 1. 驱动预检逻辑 (Pre-flight Check)
            # ==========================================
            selection = GraphicsPipeSelection.get_global_ptr()
            logger.info("SIM: 正在扫描系统显卡驱动管道...")
            
            supported_pipes = []
            for i in range(selection.get_num_pipe_types()):
                pipe_type = selection.get_pipe_type(i)
                supported_pipes.append(pipe_type.name)
            
            logger.info(f"SIM: 系统可用渲染管道: {supported_pipes}")
            
            # 尝试创建默认管道，验证驱动是否真的能跑通
            temp_pipe = selection.make_default_pipe()
            if temp_pipe is None:
                error_msg = (
                    "!!! 关键错误: 无法创建图形管道 (Graphics Pipe) !!!\n"
                    "原因分析: \n"
                    "1. 物理显卡驱动 (NVIDIA/AMD) 未正确安装或损坏。\n"
                    "2. 缺少必要库 (libEGL.so, libGL.so)。\n"
                    "3. 环境变量 LD_LIBRARY_PATH 未指向驱动路径。\n"
                    "由于你已禁用 tinydisplay，程序将立即终止。"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.success(f"SIM: 成功连接 GPU 驱动: {temp_pipe.get_interface_name()}")
            # ==========================================
            if cls._debug_mode <= DEBUG_MODE.INFO:
                cls.load_config("gl-debug", "#t") # 启用 OpenGL 的调试模式
                cls.load_config("want-pstats", "1") # 启用 Panda3D 的性能统计工具 PStats
            
            # 彻底强制采用纯无头 GPU 渲染 (EGL)
            cls.load_config("load-display", "egl")
            cls.load_config("window-type", "offscreen")
            cls.load_config("gl-debug", "#f")
            
            # 设置辅助渲染器
            aux_displays = [
                "p3headlessgl"
            ]

            for display in aux_displays:
                cls.load_config("aux-display", display)

            # Load other configurations
            configs = {
                "gl-version": "3 3",
                "sync-video": "false", # 禁用垂直同步，否则渲染速率会被限制为屏幕的刷新率
                "model-cache-compressed-textures": "1", # 启用模型缓存中的压缩纹理，这可以减少内存使用，提高性能。
                # 【提速点1】降低或关闭抗锯齿，大幅减少多相机时的显存带宽占用
                "framebuffer-multisample": "0", 
                "multisamples": "0", 
                # 【提速点2】告诉引擎火力全开，禁止帧间休眠和线程让出阻塞
                "yield-timeslice": "false",
                "client-sleep": "0",
                "audio-library-name": "null", # 禁用音频库，不处理音频输出
                "notify-level": cls._debug_mode.name.lower(), # 设置通知级别
                "default-directnotify-level": cls._debug_mode.name.lower(), # 设置默认的直接通知级别
                "print-pipe-types": "false", # 禁止打印管道类型信息
                # "show-buffers": "#t", # 开启 Panda3D 的缓冲区可视化功能
                "threading-model": "Cull/Draw" # 设置 Panda3D 的线程模型。这里指定使用分离的剔除（Cull）和绘制（Draw）线程，这样可以在多核处理器上提高渲染效率。
            }
            for key, value in configs.items():
                cls.load_config(key, value)
                
        it = cls.__dict__.get("__it__")
        if it is None:
            cls.__it__ = it = object.__new__(cls)
            it.init()
        return it

    def __init__(self) -> None:
        """单例模式 (singleton pattern), 使用 init() 而不是这里的 __init__()
        """
        pass

    def init(self) -> None:
        """Initializer for the purposes of maintaining a singleton of this class.
        """
        self._render_lock = Lock()
        try:
            # There can be only 1 ShowBase instance at a time.
            if _ShowBaseInstance._render_mode == "offscreen":
                super().__init__(windowType="offscreen") # 此时是没有界面的
            elif _ShowBaseInstance._render_mode == "onscreen":
                super().__init__() # 开启可视化界面
            simplepbr.init(
                msaa_samples=0, # 【提速点1-补充】将 PBR 原本极高的 16 倍抗锯齿设为0（极致性能），BEV不需要极高抗锯齿
                use_hardware_skinning=True,
                use_normal_maps=True,
                use_330=False
            ) # https://github.com/Moguri/panda3d-simplepbr

            self.setBackgroundColor(255, 255, 255, 1) # 设置背景颜色, (0,0,0) 是黑色
            self.setFrameRateMeter(True) # 是否显示 FPS
            logger.info("SIM: 初始化 ShowBase 实例")
        except Exception as e:
            raise e
        
    # #################################
    # 下面两个 method 用于调整 class 的参数
    # #################################
    @classmethod
    def set_render_mode(cls, render_mode: str) -> None:
        """Sets the render mode.
        """
        cls._render_mode = render_mode
              
    @classmethod
    def set_rendering_verbosity(cls, debug_mode: DEBUG_MODE) -> None:
        """Set rendering debug information verbosity.
        """
        cls._debug_mode = debug_mode
        cls.load_config("notify-level", cls._debug_mode.name.lower())
        cls.load_config("default-directnotify-level", cls._debug_mode.name.lower())

    @classmethod
    def set_rendering_backend(
        cls,
        rendering_backend: BACKEND_LITERALS,
    ) -> None:
        """Sets the rendering backend.
        """
        if "__it__" not in cls.__dict__:
            cls._rendering_backend = rendering_backend
        else:
            if cls._rendering_backend != rendering_backend:
                logger.warning("SIM: Cannot apply rendering backend after setup.")

    # ##################
    # 关于 showbase 的删除
    # ##################
    def destroy(self) -> None:
        """Destroy this renderer and clean up all remaining resources.
        """
        super().destroy()
        self.__class__.__it__ = None

    def __del__(self) -> None:
        try:
            self.destroy()
        except (AttributeError, TypeError):
            pass
    
    # #############
    # 设置 SIM ROOT
    # #############
    def setup_sim_root(self, simid: str):
        """Creates the simulation root node in the scene graph.
        """
        root_np = NodePath(simid)
        # 根节点放在 render 上面
        with self._render_lock:
            root_np.reparentTo(self.render)
                    
        unlit_shader = Shader.load(
            Shader.SL_GLSL,
            vertex=current_file_path("../_assets_3d/shader/unlit_shader.vert"),
            fragment=current_file_path("../_assets_3d/shader/unlit_shader.frag"),
        )
        root_np.setShader(unlit_shader, priority=10)

        logger.info("SIM: 完成了 SIM Root 的设置.")
        return root_np