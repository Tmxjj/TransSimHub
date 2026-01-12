'''
@Author: WANG Maonan
@Date: 2024-07-13 07:26:57
@Description: 生成 map glb 文件
LastEditTime: 2026-01-11 20:12:39
'''
import math
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh.visual
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from . import OLD_TRIMESH
from ...vis3d_utils.colors import Colors
from ...vis3d_utils.coordinates import BoundingBox
from ..sumonet_convert_utils.glb_data import GLBData
from ..sumonet_convert_utils.geometry import triangulate_polygon

# Suppress trimesh deprecation warning
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.",
        category=DeprecationWarning,
    )
    import trimesh  # only suppress the warnings caused by trimesh
    from trimesh.exchange import gltf
    from trimesh.visual.material import PBRMaterial


def make_map_glb(
    polygons: List[Tuple[Polygon, Dict[str, Any]]],
    bbox: BoundingBox,
    lane_dividers,
    edge_dividers,
) -> GLBData:
    """Create a GLB file from a list of road polygons.
    """
    # Attach additional information for rendering as metadata in the map glb
    metadata = {
        "bounding_box": (
            bbox.min_pt.x,
            bbox.min_pt.y,
            bbox.max_pt.x,
            bbox.max_pt.y,
        ),
        "lane_dividers": lane_dividers,
        "edge_dividers": edge_dividers,
    }
    scene = trimesh.Scene(metadata=metadata)

    normal_polygons = []
    junction_groups = {} # Key: JunctionID, Value: List of (Polygon, Metadata)

    for poly, meta in polygons:
        road_id = str(meta.get("road_id", ""))
        
        # 判断是否为内部车道 (SUMO中内部车道ID通常以 ':' 开头)
        if road_id.startswith(":"):
            # 推断 Junction ID。通常格式为 :JunctionID_Index
            # 我们通过去掉最后一个下划线及其后的部分来聚合属于同一交叉口的车道
            # 例如 :intersection_1_1_0 -> :intersection_1_1
            junction_id = road_id.rsplit('_', 1)[0]
            
            if junction_id not in junction_groups:
                junction_groups[junction_id] = []
            junction_groups[junction_id].append((poly, meta))
        else:
            # 普通道路直接保留
            normal_polygons.append((poly, meta))

    processed_polygons = list(normal_polygons)

    # 对每个交叉口组进行几何合并
    for j_id, items in junction_groups.items():
        polys_to_merge = [p for p, m in items]
        
        # 合并多边形
        merged_geom = unary_union(polys_to_merge)
        
        # 准备元数据 (使用组内第一个元素的元数据，但修改 ID)
        base_meta = items[0][1].copy()
        base_meta["road_id"] = j_id
        if "lane_id" in base_meta:
            del base_meta["lane_id"] # 合并后不再有单一车道ID

        # unary_union 可能返回 Polygon 或 MultiPolygon
        if isinstance(merged_geom, Polygon):
            if not merged_geom.is_empty:
                processed_polygons.append((merged_geom, base_meta))
        elif isinstance(merged_geom, MultiPolygon):
            for geom in merged_geom.geoms:
                if not geom.is_empty:
                    processed_polygons.append((geom, base_meta))
    
    # 使用处理后的多边形列表生成 Mesh
    meshes = _generate_meshes_from_polygons(processed_polygons)
    
    material = PBRMaterial(
        name="RoadDefault",
        baseColorFactor=Colors.DarkGrey.value,
        metallicFactor=0.8, # 高金属感
        roughnessFactor=0.8, # 低粗糙度，更光滑
    )

    for mesh in meshes:
        mesh.visual.material = material
        road_id = mesh.metadata["road_id"]
        lane_id = mesh.metadata.get("lane_id")
        name = str(road_id)
        if lane_id is not None:
            name += f"-{lane_id}"
        if OLD_TRIMESH:
            scene.add_geometry(mesh, name, extras=mesh.metadata)
        else:
            scene.add_geometry(mesh, name, geom_name=name, metadata=mesh.metadata)
    return GLBData(gltf.export_glb(scene, include_normals=True))

def _generate_meshes_from_polygons(
    polygons: List[Tuple[Polygon, Dict[str, Any]]],
) -> List[trimesh.Trimesh]:
    """Creates a mesh out of a list of polygons.
    """
    meshes = []

    # Trimesh's API require a list of vertices and a list of faces, where each
    # face contains three indexes into the vertices list. Ideally, the vertices
    # are all unique and the faces list references the same indexes as needed.
    for poly, metadata in polygons:
        vertices, faces = [], []
        point_dict = dict()
        current_point_index = 0

        # Collect all the points on the shape to reduce checks by 3 times
        for x, y in poly.exterior.coords:
            p = (x, y, 0)
            if p not in point_dict:
                vertices.append(p)
                point_dict[p] = current_point_index
                current_point_index += 1
        triangles = triangulate_polygon(poly)
        for triangle in triangles:
            face = np.array(
                [point_dict.get((x, y, 0), -1) for x, y in triangle.exterior.coords]
            )
            # Add face if not invalid
            if -1 not in face:
                faces.append(face)

        if not vertices or not faces:
            continue

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, metadata=metadata)

        # Trimesh doesn't support a coordinate-system="z-up" configuration, so we
        # have to apply the transformation manually.
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [-1, 0, 0])
        )
        meshes.append(mesh)
    return meshes