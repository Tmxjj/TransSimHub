'''
@Author: WANG Maonan
@Date: 2024-07-05 22:18:36
@Description: 3D 渲染过程中需要使用的颜色
LastEditTime: 2026-04-19 14:58:22
'''
from enum import Enum

# Color channel order: RGBA
class Colors(Enum):
    """Common simulation colors as RGBA values.
    """
    Red = (210 / 255, 30 / 255, 30 / 255, 1)
    Rose = (196 / 255, 0, 84 / 255, 1)
    Maroon = (128 / 255, 0, 0, 1)
    Orange = (237 / 255, 109 / 255, 0, 1)
    Yellow = (255 / 255, 190 / 255, 40 / 255, 1)
    GreenTransparent = (98 / 255, 178 / 255, 48 / 255, 0.3)
    Silver = (192 / 255, 192 / 255, 192 / 255, 1)
    Black = (0, 0, 0, 1)
    Green = (30 / 255, 210 / 255, 30 / 255, 1)

    DarkBlue = (5 / 255, 5 / 255, 70 / 255, 1)
    Blue = (0, 153 / 255, 1, 1)
    LightBlue = (173 / 255, 216 / 255, 230 / 255, 1)
    BlueTransparent = (60 / 255, 170 / 255, 200 / 255, 0.6)

    DarkCyan = (47 / 255, 79 / 255, 79 / 255, 1)
    CyanTransparent = (48 / 255, 181 / 255, 197 / 255, 0.5)

    DarkPurple = (50 / 255, 30 / 255, 50 / 255, 1)
    Purple = (127 / 255, 0, 127 / 255, 1)
    WarmAsphalt = (173 / 255, 171 / 255, 170 / 255, 1) # 温暖的柏油路颜色
    Asphalt = (140 / 255, 140 / 255, 140 / 255, 1) # 标准柏油路颜色
    LightWarmAsphalt = (118 / 255, 117 / 255, 115 / 255, 1) # 更加浅色的柏油路颜色

    DarkGrey = (80 / 255, 80 / 255, 80 / 255, 1)
    Grey = (119 / 255, 136 / 255, 153 / 255, 1)
    LightGrey = (140 / 255, 140 / 255, 140 / 255,1)
    LightGreyTransparent = (221 / 255, 221 / 255, 221 / 255, 0.1)

    OffWhite = (210 / 255, 210 / 255, 210 / 255, 1)
    White = (1, 1, 1, 1)
    # 【新增】一种偏暖的干土地颜色，亮度适中，避免过曝
    DryEarth = (135 / 255, 130 / 255, 115 / 255, 1)





class SceneColors(Enum):
    """Simulation feature colors as RGBA values
    """
    Agent = Colors.Red.value
    SocialAgent = Colors.Blue.value
    SocialVehicle = Colors.Silver.value

    Road = Colors.WarmAsphalt.value
    EgoWaypoint = Colors.CyanTransparent.value
    EgoDrivenPath = Colors.CyanTransparent.value
    BubbleLine = Colors.LightGreyTransparent.value
    MissionRoute = Colors.GreenTransparent.value
    LaneDivider = Colors.OffWhite.value
    EdgeDivider = Colors.White.value
    MedialDivider = Colors.Yellow.value
    Ground = Colors.OffWhite.value

    SignalUnknown = Colors.Grey.value
    SignalStop = Colors.Maroon.value
    SignalCaution = Colors.Yellow.value
    SignalGo = Colors.Green.value



def srgb_to_linear(srgb_color):
    """将 sRGB 空间颜色转换为 linear 空间 (供 PBR 着色器使用).

    Why:
    - SceneColors 中的数值以 sRGB 空间定义 (设计师期望在显示器上看到的显示值).
    - 启用 `framebuffer-srgb 1` 的 PBR 流水线要求 shader 输出为 linear 空间,
      OpenGL 在写入 framebuffer 时自动做 linear→sRGB 转换.
    - 若把 sRGB 数值直接当作 baseColor / vertex color 喂给 shader, 相当于
      把 linear 值再 sRGB 编码一次, 最终屏幕上的颜色会显著偏亮/偏灰, 脱离设计意图.
    - 因此在作为 baseColor / setColor 输入前, 必须先从 sRGB 逆回到 linear.

    How to apply:
    - 对主光照受体 (road / ground / 车辆 baseColor factor) 均可调用此函数.
    - 对走 `setLightOff(1)` 的线条 (车道线/边界线等), 不经过光照计算,
      一般保持 sRGB 值即可, 不必转换.
    """
    def _conv(c: float) -> float:
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4
    r, g, b, a = srgb_color
    return (_conv(r), _conv(g), _conv(b), a)