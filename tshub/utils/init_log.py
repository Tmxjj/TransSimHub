'''
@Author: WANG Maonan
@Date: 2023-08-23 11:07:43
@Description: 初始化 Log, 分为以下几个部分:
1. INFO 级别的日志打印在控制台;
2. 仿真相关的日志存储在 SIM 开头的文件
3. 算法相关的日志存储在 Traing 开头的文件
LastEditTime: 2026-01-15 10:36:20
'''
import os
import sys
from loguru import logger
from datetime import datetime

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
    """单独过滤出 Golden Dataset Generation 相关的日志
    """
    if '[GOLDEN]' in record['message']:
        return True
    return False

def rollout_filter(record) -> bool:
    """单独过滤出 rollout 枚举相关的日志
    """
    if '[ROLLOUT]' in record['message']:
        return True
    return False

def bulletin_filter(record) -> bool:
    """单独过滤出 EventBulletin 上下游广播相关的日志
    """
    if '[Bulletin]' in record['message']:
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
    now = datetime.strftime(datetime.now(),'%Y-%m-%d_%H_%M_%S')
    log_path = os.path.join(log_path, now)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    # Remove default logger
    logger.remove()

    logger.add(
        os.path.join(log_path, './SIM-{time}.log'), 
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}", 
        filter=simulation_filter, 
        level=file_log_level, 
        rotation="7 MB"
    )

    logger.add(
        os.path.join(log_path, './Traing-{time}.log'), 
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}", 
        filter=training_filter, 
        level=file_log_level, 
        rotation="7 MB"
    )

    #  通用/评估日志文件 (捕获除 SIM 和 RL 以外的所有日志)
    logger.add(
        os.path.join(log_path, './Eval-{time}.log'), 
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}", 
        filter=evaluation_filter, 
        level=file_log_level, 
        rotation="7 MB"
    )

    # [新增] 配置日志文件 (捕获所有 [CFG] 开头的日志)
    logger.add(
        os.path.join(log_path, './CFG-{time}.log'), 
        format="{time} | {level:<6} | {message}", # 简化格式，配置日志不需要行号
        filter=config_filter, 
        level=file_log_level, 
        rotation="1 MB"
    )

    # [新增] Golden Generation 日志文件 (捕获所有 [GOLDEN] 开头的日志)
    logger.add(
        os.path.join(log_path, './Golden-{time}.log'),
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}",
        filter=golden_filter,
        level=file_log_level,
        rotation="7 MB"
    )

    # [新增] Rollout 日志文件 (捕获所有 [ROLLOUT] 开头的日志)
    logger.add(
        os.path.join(log_path, './Rollout-{time}.log'),
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}",
        filter=rollout_filter,
        level=file_log_level,
        rotation="7 MB"
    )

    # [新增] Bulletin 广播日志文件 (捕获所有 [Bulletin] 开头的日志)
    logger.add(
        os.path.join(log_path, './Bulletin-{time}.log'),
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}",
        filter=bulletin_filter,
        level=file_log_level,
        rotation="3 MB"
    )

    # [新增] VLM 推理决策日志文件 (捕获 [VLM]、[MaxPressure]、[FixedTime] 日志)
    logger.add(
        os.path.join(log_path, './VLM-{time}.log'),
        format="{time} | {level:<6} | {name}:{function}:{line} - {message}",
        filter=vlm_filter,
        level=file_log_level,
        rotation="7 MB"
    )

    # Terminal handler
    logger.add(
        sys.stderr, 
        format="'<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'", 
        level=terminal_log_level
    )