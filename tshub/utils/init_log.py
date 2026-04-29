'''
@Author: WANG Maonan
@Date: 2023-08-23 11:07:43
@Description: 初始化 Log, 分为以下几个部分:
1. INFO 级别的日志打印在控制台;
2. 仿真相关的日志存储在 SIM 开头的文件
3. 算法相关的日志存储在 Traing 开头的文件
LastEditTime: 2026-04-28 23:59:45
'''
import os
import sys
from loguru import logger
from datetime import datetime


def _build_log_file_map(log_path: str, session_tag: str) -> dict:
    """根据固定的 session_tag 构造本次运行使用的日志文件路径。

    之所以单独抽出来，是为了让主进程与 spawn 出来的子进程能够共享完全相同的一组文件名。
    如果子进程继续使用 `Golden-{time}.log` 这种动态文件名，它会在自己初始化 logger 的时刻
    再生成一套新文件，最终导致 worker 日志无法并入主进程那份 Golden 日志。
    """
    return {
        "sim": os.path.join(log_path, f"SIM-{session_tag}.log"),
        "training": os.path.join(log_path, f"Traing-{session_tag}.log"),
        "eval": os.path.join(log_path, f"Eval-{session_tag}.log"),
        "cfg": os.path.join(log_path, f"CFG-{session_tag}.log"),
        "golden": os.path.join(log_path, f"Golden-{session_tag}.log"),
        "vlm": os.path.join(log_path, f"VLM-{session_tag}.log"),
    }


def _configure_logger_handlers(
    log_path: str,
    session_tag: str,
    file_log_level: str = "DEBUG",
    terminal_log_level: str = 'INFO',
    remove_existing: bool = True,
    include_terminal: bool = True,
):
    """按给定的固定文件名配置 logger handlers。"""
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    if remove_existing:
        logger.remove()

    log_files = _build_log_file_map(log_path, session_tag)

    logger.add(
        log_files["sim"],
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}",
        filter=simulation_filter,
        level=file_log_level,
        rotation="7 MB"
    )

    logger.add(
        log_files["training"],
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}",
        filter=training_filter,
        level=file_log_level,
        rotation="7 MB"
    )

    logger.add(
        log_files["eval"],
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}",
        filter=evaluation_filter,
        level=file_log_level,
        rotation="7 MB"
    )

    logger.add(
        log_files["cfg"],
        format="{time} | {level:<6} | {message}",
        filter=config_filter,
        level=file_log_level,
        rotation="1 MB"
    )

    logger.add(
        log_files["golden"],
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}",
        filter=golden_filter,
        level=file_log_level,
        rotation="7 MB"
    )

    logger.add(
        log_files["vlm"],
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}",
        filter=vlm_filter,
        level=file_log_level,
        rotation="7 MB"
    )

    if include_terminal:
        logger.add(
            sys.stderr,
            format="'<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'",
            level=terminal_log_level
        )

    return log_files

def simulation_filter(record) -> bool:
    """单独过滤出仿真部分产生的日志

    Args:
        record (_type_): _description_
    """
    if 'SIM' in record['message']:
        return True
    return False


def training_filter(record) -> bool:
    """单独过滤出训练部分的日志

    Args:
        record (_type_): _description_

    Returns:
        bool: _description_
    """
    if 'RL' in record['message']:
        return True
    return False

def evaluation_filter(record) -> bool:
    """单独过滤出评估过程中的常规日志 (排除 SIM 和 RL)
    """
    # 包含 [EVAL] 或者 [CFG] 的日志都归类到 Evaluator 日志
    if '[EVAL]' in record['message']:
        return True
    return False

def config_filter(record) -> bool:
    """单独过滤出配置相关的日志
    """
    if '[CFG]' in record['message']:
        return True
    return False

def golden_filter(record) -> bool:
    """过滤出 Golden 数据生成相关日志，并将 rollout/worker/bulletin 合并到同一份日志中。

    约定：
      - `Golden-{time}.log` 作为 Golden 数据生成流程的统一日志文件；
      - `[GOLDEN]` / `[Golden]` 记录主流程日志；
      - `[ROLLOUT]` / `[ROLLOUT-WORKER]` 记录并行 rollout 相关日志；
      - `[Bulletin]` 记录上下游协同广播日志；
      - 历史上的 `[DIAG]` 日志已经废弃，若还有残留，也并回 Golden 日志，避免额外生成 DIAG 文件。
    """
    msg = record['message']
    if (
        '[GOLDEN]' in msg
        or '[Golden]' in msg
        or '[ROLLOUT]' in msg
        or '[ROLLOUT-WORKER]' in msg
        or '[Bulletin]' in msg
        or '[DIAG]' in msg
    ):
        return True
    return False

def vlm_filter(record) -> bool:
    """单独过滤出 VLM 推理决策相关的日志（[VLM]、[MaxPressure]、[FixedTime]）
    """
    msg = record['message']
    if '[VLM]' in msg or '[MaxPressure]' in msg or '[FixedTime]' in msg:
        return True
    return False

def set_logger(log_path, file_log_level="DEBUG", terminal_log_level='INFO'):
    session_tag = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S_%f')
    log_path = os.path.join(log_path, session_tag)
    _configure_logger_handlers(
        log_path=log_path,
        session_tag=session_tag,
        file_log_level=file_log_level,
        terminal_log_level=terminal_log_level,
        remove_existing=True,
        include_terminal=True,
    )
    return {
        "log_dir": log_path,
        "session_tag": session_tag,
        "log_files": _build_log_file_map(log_path, session_tag),
    }


def attach_logger_to_existing_session(
    log_dir: str,
    session_tag: str,
    file_log_level: str = "DEBUG",
    terminal_log_level: str = 'INFO',
    include_terminal: bool = False,
):
    """让新进程挂载到已有日志会话。

    典型用法是：
      - 主进程调用 `set_logger()` 创建一套新的日志目录与文件；
      - spawn 出来的 worker 子进程调用本函数，复用同一目录与同一批文件名。
    """
    _configure_logger_handlers(
        log_path=log_dir,
        session_tag=session_tag,
        file_log_level=file_log_level,
        terminal_log_level=terminal_log_level,
        remove_existing=True,
        include_terminal=include_terminal,
    )
    return _build_log_file_map(log_dir, session_tag)
