'''
@Author: WANG Maonan
@Date: 2024-07-15 11:53:11
@Description: 创建不同类型的 Offscreen Camera Type
LastEditTime: 2026-04-18 19:27:34
'''
from panda3d.core import (
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    OrthographicLens,
    Texture,
    WindowProperties,
    PerspectiveLens,
)

from .offscreen_camera import (
    # 俯拍摄像头
    OffscreenBEVCamera,
    # 路口摄像头
    OffscreenJunctionFrontCamera,
    OffscreenJunctionBackCamera,
    # 前方摄像头
    OffscreenFrontCamera,
    OffscreenFrontRightCamera,
    OffscreenFrontLeftCamera,
    # 后方摄像头
    OffscreenBackCamera,
    OffscreenBackLeftCamera,
    OffscreenBackRightCamera,
    # 无人机的摄像头
    OffscreenAircraftCamera,
)
from .offscreen_camera.offscreen_camera_type import OffscreenCameraType

def build_offscreen_camera(
    name: str, # camera_id
    mask,
    width: int,
    height: int,
    resolution: float,
    showbase_instance,
    root_np, 
    camera_type: str = 'Off_BEV_Camera' # 默认为 BEV
) -> None:
    """生成一个 offscreen 的 camera. 每一个 camera 都会绑定在一个 sensor 上面, 然后 P3DOffscreenCamera 会设置角度

    Args:
        name (str): camera 的 id, 用于创建 node, 和找到这个 camera
        width (int): 生成的图像的宽度
        height (int): 生成的图像的高度
        resolution (float): 缩放因子，它用于缩放胶片大小的宽度和高度。例如：
            1. resolution=1 表示胶片大小被设置为原始的 width 和 height 值，没有进行任何缩放。视野直接基于这些尺寸。
            2. resolution=0.1 表示胶片大小被缩小到原始 width 和 height 值的10%。这实际上缩小了视野，使得场景中的对象看起来更大或更近，因为你是在放大观察场景的更小部分。
    """
    # setup buffer
    win_props = WindowProperties.size(width, height)
    fb_props = FrameBufferProperties()
    fb_props.setRgbColor(True)
    fb_props.setRgbaBits(8, 8, 8, 8)
    # XXX: Though we don't need the depth buffer returned, setting this to 0
    #      causes undefined behavior where the ordering of meshes is random.
    fb_props.setDepthBits(24)         # 深度缓冲位数
    fb_props.setAuxRgba(1)            # 添加辅助通道（用于阴影）
    fb_props.setStencilBits(8)        # 启用模板缓冲（某些阴影技术需要）
    
    buffer = showbase_instance.win.engine.makeOutput(
        showbase_instance.pipe,
        "{}-buffer".format(name),
        1,
        fb_props,
        win_props,
        GraphicsPipe.BFRefuseWindow,
        showbase_instance.win.getGsg(),
        showbase_instance.win,
    )
    # Set background color to black
    buffer.setClearColor((0, 0, 0, 0))

    # setup texture
    tex = Texture()
    region = buffer.getDisplayRegion(0)
    region.window.addRenderTexture(
        tex, GraphicsOutput.RTM_copy_ram, GraphicsOutput.RTP_color
    )

    # setup camera
    lens = PerspectiveLens() # 人眼的视角, 有 3D 效果
    # lens = OrthographicLens() # 这一类的 camera 没有 3D 的效果
    # 先设置 resolution=1 时的基准胶片尺寸，再用 setFov(90) 反算出固定焦距；
    # 之后用 resolution 缩放胶片尺寸 + 固定焦距，使 resolution 真正控制视野范围：
    #   resolution < 1 → 胶片缩小 → FOV 变窄 → Zoom In（放大）
    #   resolution > 1 → 胶片增大 → FOV 变宽 → Zoom Out（缩小）
    #   resolution = 1 → FOV = 90°（与原先行为一致）
    lens.setFilmSize(width, height)  # 基准胶片尺寸（resolution=1）
    lens.setFov(90)  # 基于基准胶片计算焦距: focal_length = width / (2 × tan(45°))
    base_focal_length = lens.getFocalLength()  # 取出固定焦距
    lens.setFilmSize(width * resolution, height * resolution)  # 用 resolution 缩放胶片
    lens.setFocalLength(base_focal_length)  # 锁定焦距，让胶片大小决定实际 FOV

    camera_np = showbase_instance.makeCamera(
        buffer, camName=name, 
        scene=root_np, lens=lens
    )
    camera_np.reparentTo(root_np) # 设置 camera 在 node 上

    # mask is set to make undesirable objects invisible to this camera
    camera_np.node().setCameraMask(mask)

    # #########################################
    # 设置 camera, 这里 camera update 的方式不一样
    # #########################################
    
    # 跟车的视角
    _camera_type = OffscreenCameraType(camera_type)
    if _camera_type == OffscreenCameraType.BEV:
        camera = OffscreenBEVCamera(camera_np=camera_np, buffer=buffer, tex=tex, showbase_instance=showbase_instance)
    # 路口的摄像头
    elif _camera_type == OffscreenCameraType.Junction_Front: # 正对路口
        camera = OffscreenJunctionFrontCamera(camera_np, buffer, tex, showbase_instance)
    elif _camera_type == OffscreenCameraType.Junction_Back: # 对着道路出口
        camera = OffscreenJunctionBackCamera(camera_np, buffer, tex, showbase_instance)
    # 前拍
    elif _camera_type == OffscreenCameraType.Front:
        camera = OffscreenFrontCamera(camera_np, buffer, tex, showbase_instance)
    elif _camera_type == OffscreenCameraType.Front_LEFT: # 前拍 (左侧)
        camera = OffscreenFrontLeftCamera(camera_np, buffer, tex, showbase_instance)
    elif _camera_type == OffscreenCameraType.Front_RIGHT: # 前拍 (右侧)
        camera = OffscreenFrontRightCamera(camera_np, buffer, tex, showbase_instance)
    # 后拍
    elif _camera_type == OffscreenCameraType.Back:
        camera = OffscreenBackCamera(camera_np, buffer, tex, showbase_instance)
    elif _camera_type == OffscreenCameraType.Back_LEFT: # 后拍 (左侧)
        camera = OffscreenBackLeftCamera(camera_np, buffer, tex, showbase_instance)
    elif _camera_type == OffscreenCameraType.Back_RIGHT: # 后拍 (右侧)
        camera = OffscreenBackRightCamera(camera_np, buffer, tex, showbase_instance)
    # 无人机的视角
    elif _camera_type == OffscreenCameraType.Aircraft: # 无人机从上往下拍摄
        camera = OffscreenAircraftCamera(camera_np, buffer, tex, showbase_instance)  
    else:
        raise ValueError(f"请你确认 camera 的名字, 没有 {camera_type}.")
    return camera