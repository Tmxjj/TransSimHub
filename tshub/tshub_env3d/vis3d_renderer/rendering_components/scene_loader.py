'''
@Author: WANG Maonan
@Date: 2024-07-12 21:38:26
@Description: 场景加载相关的方法 (用于初始化场景)
LastEditTime: 2026-04-19 16:54:54
'''
from ....utils.get_abs_path import get_abs_path
current_file_path = get_abs_path(__file__)
import os
import simplepbr
from pathlib import Path
from loguru import logger
from panda3d.core import (
    AntialiasAttrib,
    Shader,
    SamplerState,
    ShaderTerrainMesh,
    Geom,
    GeomLinestrips,
    GeomNode,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexReader,
    GeomVertexWriter,
    AmbientLight,
    CardMaker,
    Material,
    Vec3,
    Vec4,
    DirectionalLight,
    loadPrcFileData
)
from ...vis3d_utils.masks import CamMask
from ...vis3d_utils.colors import SceneColors, Colors, srgb_to_linear

class SceneLoader(object):
    ROAD_MAP_NODE_NAME = "road_map"
    MAP_FILENAME = "map.glb"
    TERRAIN_FILENAME = "ground.glb"
    LANE_FILENAME = "lane_lines.glb"
    ROAD_FILENAME = "road_lines_wo_middle.glb" # 边缘线（白）
    MIDDLE_FILENAME = "middle_lines.glb"      # 中央分隔带（黄）
    TURN_FILENAME = "turn.glb"

    def __init__(
            self, 
            root_np,
            showbase_instance,
            skybox_dir:str,
            terrain_dir:str,
            scenario_glb_dir:str,
            map_road_lane_glsl_dir:str
        ) -> None:
        self.skybox_dir = Path(skybox_dir) # the skybox path model
        self.terrain_dir = Path(terrain_dir) # the path for terrain model
        self.scenario_glb_dir = Path(scenario_glb_dir) # 场景地图的路径
        self.map_road_lane_glsl_dir = Path(map_road_lane_glsl_dir) # lane 的 glsl 文件
        self._showbase_instance = showbase_instance # panda3d ShowBase (所有模型都挂载在这上面)
        self._root_np = root_np # 用于挂载 node

        # load map 之后场景的基础信息
        self.map_radius = None
        self.map_center = None
    

    def initialize_scene(self) -> None:
        logger.info("SIM: Starting TSHub3D scene initialization.")
        # 开启场景抗锯齿   
        self._root_np.set_antialias(AntialiasAttrib.MAuto)

        self.load_map()
        # --- 分别加载白色边缘线和黄色中央线 ---
        wo_middle_path = self.scenario_glb_dir / SceneLoader.ROAD_FILENAME
        if wo_middle_path.exists():
            self.load_road_lines_white(filename=SceneLoader.ROAD_FILENAME)
            self.load_road_lines_yellow()
        else:
            self.load_road_lines_white(filename="road_lines.glb")

        self.load_lane_lines()
        self.load_turn_marking()
        self.load_flat_terrain()
        self.load_sky_box()
        self.setup_lighting()

        return self.map_radius, self.map_center

    def load_map(self) -> None:
        """Load map & 并获得中心位置
        """
        map_path = self.scenario_glb_dir / self.MAP_FILENAME
        logger.info(f"SIM: 加载场景地图, {map_path}.")
        try:
            map_np = self._showbase_instance.loader.loadModel(map_path, noCache=True)
            node_path = self._root_np.attachNewNode(self.ROAD_MAP_NODE_NAME)
            map_np.reparent_to(node_path)
            # node_path.setDepthOffset(0)
            # 定义 mask
            node_path.hide(CamMask.AllOn)
            node_path.show(CamMask.MapMask) # 只给部分 camera 展示
            # 设置路面的颜色: 先把 sRGB 设计色转成 linear, 再喂给 PBR shader
            # 搭配 framebuffer-srgb=1 在输出阶段自动 linear→sRGB, 最终显示回设计色 (0.314 DarkGrey)
            node_path.setColor(srgb_to_linear(SceneColors.Road.value), 1)

            # 如果原来的路面有纹理，setColor 只会变成“染色”。如果你想完全替换成纯色，需要把纹理和材质关掉 （直接修改glb会更好）
            node_path.setTextureOff(1)  # 强制移除贴图
            # 🔴 深层清理 shader: map.glb 是一整颗子树, 每个 GeomNode 自带 glTF shader,
            #    只在顶层 node_path 上 clearShader 无法让内部 GeomNode 继承 root_np 的 simplepbr shader,
            #    导致路面走的是不采样 shadow map 的 GLB 原生 shader → 车辆投影在路面上消失。
            #    findAllMatches("**") 对子树所有节点都执行 clearShader, 强制整棵子树走 simplepbr.
            node_path.clearShader()
            for child in node_path.findAllMatches("**"):
                child.clearShader()
            node_path.setTransparency(False)  # 🔴 强制关闭透明度
            node_path.set_depth_write(True)   # 🔴 必须为 True，否则接不住阴影
            # 路面 PBR 材质 —— 非金属 + 中高粗糙度, 让阴影以漫反射亮度差清晰显现
            road_material = Material("road_material")
            road_material.setBaseColor(Vec4(1.0, 1.0, 1.0, 1.0)) # baseColor 留白, 由 setColor 染色
            road_material.setMetallic(0.0)   # 柏油路非金属
            road_material.setRoughness(0.85) # 粗糙度高, 漫反射为主, 无镜面高光
            node_path.setMaterial(road_material, 1)


            map_bounds = map_np.getBounds()
            self.map_radius = map_bounds.getRadius()
            map_model_center = map_bounds.getCenter()
            self.map_center = (
                map_model_center.getX(), 
                map_model_center.getY(), 
                map_model_center.getZ()
            )
            logger.info(f"SIM: 场景地图加载成功.")
            logger.info(f"SIM: 地图的中心 {self.map_center}.")
            logger.info(f"SIM: 地图的半径 {self.map_radius}.")
        except Exception as e:
            print(f"Error loading map: {e}")
        return map_np


    def load_road_lines_white(self, filename="road_lines_wo_middle.glb"):
        """加载道路边界线 (通常为实线，白色)"""
        road_lines_path = self.scenario_glb_dir / filename
        logger.info(f"SIM: 加载道路边界线(白), {road_lines_path}.")
        if road_lines_path.exists():
            # 修改点2：节点名设为 road_edge
            road_lines_node = self._load_line_data(road_lines_path, "road_edge")
            edge_lines_np = self._root_np.attachNewNode(road_lines_node)

            edge_lines_np.setDepthOffset(1) 
            edge_lines_np.setBin("fixed", 10)
            
            edge_lines_np.hide(CamMask.AllOn)
            edge_lines_np.show(CamMask.MapMask)
            
            # 设置边缘线颜色（白色）
            edge_lines_np.setColor(SceneColors.EdgeDivider.value)
            edge_lines_np.setRenderModeThickness(2)
            # edge_lines_np.setLightOff(1)
            edge_lines_np.set_depth_write(False)
            logger.info(f"SIM: 加载道路边界线成功.")
            return edge_lines_np
        return None
    
    def load_road_lines_yellow(self):
        """加载道路中央分隔带 (通常为黄色)"""
        road_middle_path = self.scenario_glb_dir / SceneLoader.MIDDLE_FILENAME
        logger.info(f"SIM: 加载道路中央分隔带(黄), {road_middle_path}.")
        if road_middle_path.exists():
            # 修改点3：节点名设为 road_middle，确保独立性
            road_middle_node = self._load_line_data(road_middle_path, "road_middle")
            middle_lines_np = self._root_np.attachNewNode(road_middle_node)
            
            middle_lines_np.setDepthOffset(1) 
            middle_lines_np.setBin("fixed", 10)
            
            middle_lines_np.hide(CamMask.AllOn)
            middle_lines_np.show(CamMask.MapMask)
            
            # 设置中央线颜色（黄色）
            # 注意：请确保 SceneColors.MedialDivider.value 已被定义为亮黄色 (1, 1, 0, 1)
            middle_lines_np.setColor(SceneColors.MedialDivider.value)
            middle_lines_np.setRenderModeThickness(2)
            # middle_lines_np.setLightOff(1)
            middle_lines_np.set_depth_write(False)
            logger.info(f"SIM: 加载中央分隔带成功.")
            return middle_lines_np
        return None

    def load_lane_lines(self):
        """Lane lines (dashed, white)
        """
        lane_lines_path = self.scenario_glb_dir / SceneLoader.LANE_FILENAME
        logger.info(f"SIM: 加载车道线, {lane_lines_path}.")
        if lane_lines_path.exists():
            lane_lines_np = self._load_line_data(lane_lines_path, "lane_lines")
            dashed_lines_np = self._root_np.attachNewNode(lane_lines_np)

            # 设置深度偏移：数值越大，视觉上越靠前。1 是常用的起步值。这强制实线在路面之上渲染
            dashed_lines_np.setDepthOffset(1) 
            # 将其放在 Fixed Bin 或 Transparent Bin，确保它在不透明物体(路面)之后绘制
            dashed_lines_np.setBin("fixed", 10)

            # 定义 mask
            dashed_lines_np.hide(CamMask.AllOn)
            dashed_lines_np.show(CamMask.MapMask) # 只给部分 camera 展示
            # 设置车道线的颜色
            dashed_lines_np.setColor(SceneColors.LaneDivider.value)
            dashed_lines_np.setRenderModeThickness(2)
            
            dashed_line_shader = Shader.load(
                Shader.SL_GLSL,
                vertex=self.map_road_lane_glsl_dir/"dashed_line_shader.vert",
                fragment=self.map_road_lane_glsl_dir/"dashed_line_shader.frag",
            )
            dashed_lines_np.setShader(dashed_line_shader, priority=20)
            dashed_lines_np.setShaderInput(
                "iResolution", self._showbase_instance.getSize()
            )
            # dashed_lines_np.setLightOff(1)  # 车道线不参与光照/阴影

            # 关闭深度写入：防止产生阴影
            dashed_lines_np.set_depth_write(False)
            
            logger.info(f"SIM: 加载车道线成功.")
            return dashed_lines_np


    def _load_line_data(self, path: Path, name: str) -> GeomNode:
        """从模型中提取几何线段数据，并将这些数据重新组装成一个新的 GeomNode 对象（方便后续操作）

        Args:
            path (Path): 模型的文件路径
            name (str): geom 的名称
        """
        lines = []
        road_lines_np = self._showbase_instance.loader.loadModel(path, noCache=True)
        geomNodeCollection = road_lines_np.findAllMatches("**/+GeomNode")
        for nodePath in geomNodeCollection:
            geomNode = nodePath.node()
            geom = geomNode.getGeom(0)
            vdata = geom.getVertexData()
            vreader = GeomVertexReader(vdata, "vertex")
            pts = []
            while not vreader.isAtEnd():
                v = vreader.getData3()
                pts.append((v.x, v.y, v.z))
            lines.append(pts)

        # Create geometry node
        geo_format = GeomVertexFormat.getV3()
        vdata = GeomVertexData(name, geo_format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")

        prim = GeomLinestrips(Geom.UHStatic)
        for pts in lines:
            for x, y, z in pts:
                vertex.addData3(x, y, z)
            prim.add_next_vertices(len(pts))
            assert prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        node_path = GeomNode(name)
        node_path.addGeom(geom)
        return node_path

    def load_turn_marking(self):
        """Load intersection turn markings (left/straight/right) from GLB.

        Expectations:
        - The `turn.glb` is authored in the same map coordinate frame.
        - Geometry is flat and placed slightly above the road surface to avoid z-fighting.
        Rendering policy:
        - Render after the road and lane lines via higher fixed bin.
        - Disable lighting/shader to keep colors consistent and avoid shadows.
        - Do not write depth to prevent occluding other scene elements.
        """
        turn_path = self.scenario_glb_dir / SceneLoader.TURN_FILENAME
        logger.info(f"SIM: 加载路口转向标记, {turn_path}.")
        if turn_path.exists():
            try:
                turn_np = self._showbase_instance.loader.loadModel(turn_path, noCache=True)
                node_path = self._root_np.attachNewNode("turn_marking")
                turn_np.reparent_to(node_path)

                # 显示在地图相机中
                node_path.hide(CamMask.AllOn)
                node_path.show(CamMask.MapMask)

                # 提升到路面和车道线之上渲染
                node_path.setDepthOffset(2)
                node_path.setBin("fixed", 15)

                # 关闭光照与着色，避免受阴影/高光影响；不写深度避免遮挡
                node_path.setLightOff(1)
                node_path.setShaderOff(1)
                node_path.set_depth_write(False)

                logger.info("SIM: 路口转向标记加载成功.")
                return node_path
            except Exception as e:
                logger.error(f"SIM: 路口转向标记加载失败: {e}")
        else:
            logger.warning("SIM: 未找到 turn.glb，跳过转向标记加载。")
        return None

    def setup_lighting(
            self,
            # PBR + sRGB 工作流 (配合 `framebuffer-srgb 1`, shader 内以 linear 计算):
            #   - 受光体 (road/ground) 已在 load_map/load_flat_terrain 做 sRGB→linear 转换;
            #     这里光强按"1.0 ≈ 设计参考亮度"来标定, 使受光路面亮度回到设计期望的 sRGB 数值.
            #   - ambient 略带冷色 (天光), directional 略带暖色 (阳光), 两者 ~3-5x 比例
            #     保证阴影与受光区域有明显且自然的亮度差.
            ambient_color: Vec4 = Vec4 (0.9, 0.92, 0.95, 1),      # 天光: 略偏冷
            directional_color: Vec4 = Vec4(4.5, 4.2, 3.8, 1),  # 阳光: 略偏暖, 轻度 HDR
            # 色温 API 会把 setColor 的数值整体替换为该色温下的满亮度颜色,
            # 因此要让上面 setColor 的数值真正生效, 必须保持 None.
            light_temperature: int = None,
            ambient_temperature: int = None,
            light_height: int = 100,       # (历史参数, 当前未使用; 光源 Z 由 light_direction 决定)
            light_direction: Vec3 = Vec3(-1, -1, -0.5)   # 可选光照方向
        ) -> None:
        """设置光照
        """
        logger.info("SIM: 设置光照.")

        # 确保 map_center 是 Vec3 类型
        if isinstance(self.map_center, tuple):
            map_center = Vec3(*self.map_center)  # 将 tuple 转换为 Vec3
        else:
            map_center = Vec3(self.map_center)  # 确保是 Vec3

        # 环境光
        ambient_light = AmbientLight('ambientLight')
        ambient_light.setColor(ambient_color)
        if ambient_temperature is not None:
            ambient_light.set_color_temperature(float(ambient_temperature))
        ambient_light_node_path = self._root_np.attachNewNode(ambient_light)
        self._root_np.setLight(ambient_light_node_path)

        # 定向光
        directional_light = DirectionalLight('directionalLight')
        directional_light.setColor(directional_color)
        if light_temperature is not None:
            directional_light.set_color_temperature(light_temperature)

        # 启用阴影贴图并设置覆盖范围 （注释即可关闭投影）
        # Shadow map 覆盖范围受限于 map_radius*2，过大会让每像素覆盖面积过粗、车辆阴影糊成一团；
        # 这里对大场景（如 NewYork radius~4447）做上限裁切，保证阴影清晰度。
        # 每 texel 对应的世界尺寸 = shadow_film / 8192，经验上小于 0.6m/texel 时车辆阴影边缘仍可辨认。
        SHADOW_FILM_MAX = 4096  # 上限 ~ 0.5m/texel @8192 分辨率
        shadow_film = min(self.map_radius * 2, SHADOW_FILM_MAX)
        directional_light.setShadowCaster(True, 8192, 8192) # 分辨率
        lens = directional_light.getLens()
        lens.setFilmSize(shadow_film, shadow_film) # 覆盖范围 (裁切后保证精度)

        # Panda3D 的 DirectionalLight 内置 shadow cam 默认 cameraMask = bit(31)
        directional_light.setCameraMask(CamMask.VehMask)
        directional_light_node_path = self._root_np.attachNewNode(directional_light)

        # 设置光源位置
        light_direction.normalize()

         # 计算光源位置（确保所有运算在 Vec3 上进行）
        light_pos = map_center - light_direction * self.map_radius
        light_pos.z = light_height  # 设置高度

        directional_light_node_path.setPos(light_pos)
        directional_light_node_path.lookAt(map_center)  # 朝向场景中心
        directional_light.setScene(self._root_np)
        # nearFar 按 map_radius 动态计算，避免范围过大导致 shadow_bias 等效世界偏移过大。
        # 光源在 light_height(300m) 处斜照场景，near 留足负值以捕获光源后方车辆，
        # far 按 map_radius 上限裁切保证精度。
        shadow_near = -max(self.map_radius * 0.6, 250)
        shadow_far  =  max(self.map_radius * 1.8, 500)
        lens.setNearFar(shadow_near, shadow_far)
        self._root_np.setLight(directional_light_node_path)

        if self._showbase_instance._pbr_pipeline is None:
            logger.info("SIM: 正在加载 simplepbr 着色器管线...")
            # shadow_bias 目标：world_bias ∈ [0.2m, 1.0m]
            #   world_bias = shadow_bias × (far - near)
            #   range = shadow_far - shadow_near，随 map_radius 动态变化
            #   shadow_bias = 0.3m / range，并 clamp 到 [0.00005, 0.001]
            shadow_range = shadow_far - shadow_near
            shadow_bias  = max(0.00005, min(0.001, 0.3 / shadow_range))
            logger.info(
                f"SIM: shadow nearFar=({shadow_near:.0f}, {shadow_far:.0f}), "
                f"range={shadow_range:.0f}m, bias={shadow_bias:.6f} "
                f"(world≈{shadow_bias * shadow_range:.2f}m)"
            )
            self._showbase_instance._pbr_pipeline = simplepbr.init(
                render_node=self._root_np,
                msaa_samples=16,
                use_hardware_skinning=True,
                use_normal_maps=True,
                use_occlusion_maps=True,
                use_emission_maps=True,
                use_330=True,
                enable_shadows=True,
                shadow_bias=shadow_bias,
                exposure=0.0,
            )

        
       

    def load_sky_box(self) -> None:
        """初始化环境的时候, 设置 skybox
        """
        logger.info(f"SIM: 初始化 Skybox.")
        # 加载 skybox 模型
        skybox = self._showbase_instance.loader.loadModel(self.skybox_dir/"skybox.bam")
        skybox_scale = self.map_radius * 2 # 设置 skybox 的大小
        skybox.set_scale(skybox_scale)
        # 设置 skybox 的 mask
        skybox.hide(CamMask.AllOn)
        skybox.show(CamMask.SkyBoxMask) # 只给部分 camera 展示

        # 设置 skybox 纹理
        skybox_texture = self._showbase_instance.loader.loadTexture(self.skybox_dir/"skybox.jpg")
        skybox_texture.set_minfilter(SamplerState.FT_linear)
        skybox_texture.set_magfilter(SamplerState.FT_linear)
        skybox_texture.set_wrap_u(SamplerState.WM_repeat)
        skybox_texture.set_wrap_v(SamplerState.WM_mirror)
        skybox_texture.set_anisotropic_degree(16)
        skybox.set_texture(skybox_texture)

        skybox_shader = Shader.load(
            Shader.SL_GLSL,
            self.skybox_dir/"skybox.vert.glsl",
            self.skybox_dir/"skybox.frag.glsl"
        )
        skybox.set_shader(skybox_shader)
        skybox.reparentTo(self._root_np)
        skybox.setPos(
            self.map_center[0], 
            self.map_center[1], 
            100
        )

        # Ensure the skybox is always rendered behind everything else
        skybox.set_bin('background', 0) # 确保 skybox 首先被渲染
        skybox.set_depth_write(False) # skybox 不会遮挡任意的对象
        skybox.set_compass()  # This makes the skybox fixed relative to the camera's rotation

    def load_flat_terrain(self):
        """直接加载生成的平面 terrain
        """
        ground_path = self.scenario_glb_dir / SceneLoader.TERRAIN_FILENAME
        logger.info(f"SIM: 加载地平面, {ground_path}.")
        if ground_path.exists():
            ground_np = self._showbase_instance.loader.loadModel(ground_path, noCache=True)
            node_path = self._root_np.attachNewNode("ground_node") # 在_root_np根节点下创建一个空节点（ground_node）
            ground_np.reparent_to(node_path) # 将 ground_np（地面模型的 NodePath）作为子节点附加到了 node_path 上
            
            # 稍微降低背景地面，避免与路面重叠 ---
            ground_np.setZ(-0.5) 

            # 定义 mask
            ground_np.hide(CamMask.AllOn)
            ground_np.show(CamMask.GroundMask)
            # 强制覆盖材质/纹理并着色，允许光照/阴影
            ground_np.setTextureOff(1) # 撕掉贴图（草地、水泥）
            # ground_np.setMaterialOff(1) # 撕掉材质，忽略GLB定义的粗糙或反光
            # 让地面恢复对光照和阴影的敏感度
            # ground_np.setShaderOff(1) # 不参与自动着色/阴影
            # ground_np.setLightOff(1)  # 不受灯光影响，不接收/投射阴影
            ground_np.setTransparency(False)
            # 🔴 深层清理 shader: 同 load_map 的原因, 让 ground.glb 子树继承 simplepbr shader, 正确接收阴影
            for child in ground_np.findAllMatches("**"):
                child.clearShader()
            # 保持原始设计色: SceneColors.Ground (sRGB 设计值)
            # 同样做 sRGB→linear 转换, 保证 PBR 输出回到设计色
            ground_np.setColor(srgb_to_linear(SceneColors.Ground.value), 1)
            ground_material = Material("ground_material")
            ground_material.setBaseColor(Vec4(1.0, 1.0, 1.0, 1.0))
            ground_material.setMetallic(0.0)
            ground_material.setRoughness(0.9) # 比路面更粗糙, 完全漫反射
            ground_np.setMaterial(ground_material, 1)

            ground_np.set_bin('background', 1)
            ground_np.set_depth_test(True)   # 参与深度测试
            ground_np.set_depth_write(False) # 不写深度，避免遮挡 map
            
        return ground_np