'''
@Author: WANG Maonan
@Date: 2024-07-12 21:38:26
@Description: 场景加载相关的方法 (用于初始化场景)
LastEditTime: 2026-01-08 17:41:21
'''
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
    Vec3,
    Vec4,
    DirectionalLight,
    loadPrcFileData
)
from ...vis3d_utils.masks import CamMask
from ...vis3d_utils.colors import SceneColors, Colors

class SceneLoader(object):
    ROAD_MAP_NODE_NAME = "road_map"
    MAP_FILENAME = "map.glb"
    TERRAIN_FILENAME = "ground.glb"
    LANE_FILENAME = "lane_lines.glb"
    ROAD_FILENAME = "road_lines.glb"
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
        self.load_road_lines()
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
            node_path.setDepthOffset(0)
            # 定义 mask
            node_path.hide(CamMask.AllOn)
            node_path.show(CamMask.MapMask) # 只给部分 camera 展示
            # 设置路面的颜色
            node_path.setColor(SceneColors.Road.value,1) # 修改调色 override=1 表示将原来的颜色完全覆盖

            # 如果原来的路面有纹理，setColor 只会变成“染色”。如果你想完全替换成纯色，需要把纹理和材质关掉 （直接修改glb会更好）
            # node_path.setTextureOff(1)  # 强制移除贴图
            # node_path.setMaterialOff(1) # 强制移除自带材质(防止反光/颜色干扰)

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


    def load_road_lines(self):
        """Road lines (solid, yellow)
        """
        road_lines_path = self.scenario_glb_dir / SceneLoader.ROAD_FILENAME
        logger.info(f"SIM: 加载道路边界线, {road_lines_path}.")
        if road_lines_path.exists():
            road_lines_np = self._load_line_data(road_lines_path, "road_lines")
            solid_lines_np = self._root_np.attachNewNode(road_lines_np)

            # 设置深度偏移：数值越大，视觉上越靠前。1 是常用的起步值。这强制实线在路面之上渲染
            solid_lines_np.setDepthOffset(1) 
            # 将其放在 Fixed Bin 或 Transparent Bin，确保它在不透明物体(路面)之后绘制
            solid_lines_np.setBin("fixed", 10)
            
            
            # 定义 mask
            solid_lines_np.hide(CamMask.AllOn)
            solid_lines_np.show(CamMask.MapMask) # 只给部分 camera 展示
            # 设置车道边线的颜色
            solid_lines_np.setColor(SceneColors.EdgeDivider.value)
            solid_lines_np.setRenderModeThickness(3) # 设置显示的粗细
            solid_lines_np.setLightOff(1)  # 边线不参与光照/阴影（不被照亮）
            
            # 关闭深度写入：防止产生阴影，同时解决贴地闪烁问题
            solid_lines_np.set_depth_write(False)
            logger.info(f"SIM: 加载道路线成功.")
        return solid_lines_np

    #TODO： 增加左转、知行、右转箭头的加载
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
            dashed_lines_np.setRenderModeThickness(3)
            
            dashed_line_shader = Shader.load(
                Shader.SL_GLSL,
                vertex=self.map_road_lane_glsl_dir/"dashed_line_shader.vert",
                fragment=self.map_road_lane_glsl_dir/"dashed_line_shader.frag",
            )
            dashed_lines_np.setShader(dashed_line_shader, priority=20)
            dashed_lines_np.setShaderInput(
                "iResolution", self._showbase_instance.getSize()
            )
            dashed_lines_np.setLightOff(1)  # 车道线不参与光照/阴影

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
            # ：提高环境光亮度 (0.9)，提高主光强度 (>1.0) 和暖色调
            ambient_color: Vec4 = Vec4(0.9, 0.9, 0.9, 1), 
            directional_color: Vec4 = Vec4(1.3, 1.25, 1.1, 1),  
            light_temperature: int = None,  # 关闭色温以避免偏色
            ambient_temperature: int = None,
            light_height: int = 100,
            light_direction: Vec3 = None  # 可选光照方向
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
        # directional_light.setShadowCaster(True, 2048, 2048) # 分辨率
        # lens = directional_light.getLens() 
        # lens.setFilmSize(self.map_radius * 2, self.map_radius * 2) # 覆盖范围
        # lens.setNearFar(10, self.map_radius * 3) # 深度范围

        
        directional_light_node_path = self._root_np.attachNewNode(directional_light)
        
        # 设置光源位置
        if light_direction is None:
            light_direction = Vec3(-1, -1, -0.5) 
        light_direction.normalize()
        
        # 计算光源位置（确保所有运算在 Vec3 上进行）
        light_pos = map_center - light_direction * self.map_radius
        light_pos.z = light_height  # 设置高度
        
        directional_light_node_path.setPos(light_pos)
        directional_light_node_path.lookAt(map_center)  # 朝向场景中心
        
        self._root_np.setLight(directional_light_node_path)
        self._root_np.setShaderAuto() 

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
            ground_np.show(CamMask.GroundMask) # 只给部分 camera 展示
            # 强制覆盖材质/纹理并着色，允许光照/阴影
            ground_np.setTextureOff(1) # 撕掉贴图（草地、水泥）
            # ground_np.setMaterialOff(1) # 撕掉材质，忽略GLB定义的粗糙或反光
            ground_np.setShaderOff(1) # 不参与自动着色/阴影
            ground_np.setLightOff(1)  # 不受灯光影响，不接收/投射阴影
            ground_np.setTransparency(False)
            ground_np.setColor(SceneColors.Ground.value, 1)  # 修改调色 override=1
            
            ground_np.set_bin('background', 1)
            ground_np.set_depth_test(True)   # 参与深度测试
            ground_np.set_depth_write(False) # 不写深度，避免遮挡 map
            
        return ground_np