'''
@Author: WANG Maonan
@Date: 2024-07-03 23:43:42
@Description: 继承 ShowBase, Panda3D 的主界面
LastEditTime: 2026-03-23 17:21:13
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
        configs = {
            "load-display": "pandagl",  # ✅ 换回官方默认的 OpenGL 管道
            "window-type": "offscreen", # ✅ 告诉引擎我们不需要看实体窗口
            
             # 这是标准 good practice，VirtualGL 完全支持
            "framebuffer-srgb": "1",    # 【关键修复】：强制离屏缓冲区支持 sRGB
            "double-buffer": "1",       # 必须启用
            "depth-bits": "24",         # 高质量深度
            "color-bits": "32",         # RGBA8
            "alpha-bits": "8",          # Alpha 通道
            "stencil-bits": "8",        # 🔴【非常重要！】用于裁剪阴影边缘，避免噪声
            "gl-version": "3 3",        # 🔴【非常重要！】强制开启现代 OpenGL 3.3 核心模式，原生支持 HDR
            
            "sync-video": "false",
            "model-cache-compressed-textures": "0",  # 原本为 "1"，关掉它可以避免显卡的强制有损压缩破坏模型原有质感
            "texture-anisotropic-degree": "16", # 🔥 开启 16 倍各向异性过滤，极大提升倾斜视角的纹理清晰度
            "16-bit-textures": "0",             # 禁用 16 位纹理压缩，保留 32 位全彩质感
            "dump-generated-shaders": "0",      
            "audio-library-name": "null", # 保持静音
            "print-pipe-types": "false"
        }
        
        for key, value in configs.items():
            cls.load_config(key, value)
            
        if cls._debug_mode <= DEBUG_MODE.INFO:
            cls.load_config("gl-debug", "#t")
            cls.load_config("want-pstats", "1")
            
        logger.info("SIM: 正在连接 GPU EGL 硬件渲染驱动...")
                
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
            pipeline = simplepbr.init(
                msaa_samples=16, # 改为 16 (通用最高标准支持)
                use_hardware_skinning=True,
                use_normal_maps=True,
                use_occlusion_maps=True, # 🔥 开启光照遮蔽贴图（AO贴图），让车身缝隙和底盘拥有真实的接触阴影！
                use_emission_maps=True,  # 🔥 开启自发光（可能车灯会亮）
                use_330=True,   # 开启 OpenGL 3.3 核心模式
                enable_shadows=True,
            ) # https://github.com/Moguri/panda3d-simplepbr

            if hasattr(pipeline, 'use_tonemap'):
                pipeline.use_tonemap = False
            if hasattr(pipeline, 'use_srgb'):
                pipeline.use_srgb = False

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
                    
        # unlit_shader = Shader.load(
        #     Shader.SL_GLSL,
        #     vertex=current_file_path("../_assets_3d/shader/unlit_shader.vert"),
        #     fragment=current_file_path("../_assets_3d/shader/unlit_shader.frag"),
        # )
        # root_np.setShader(unlit_shader, priority=10)

        logger.info("SIM: 完成了 SIM Root 的设置.")
        return root_np