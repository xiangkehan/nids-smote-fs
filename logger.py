"""
双语日志系统 - Bilingual Logging System

提供中英文双语日志输出，支持三种模式：
- bilingual: 同时输出中英文（默认）
- chinese: 仅中文
- english: 仅英文

使用方式:
    from logger import get_logger
    
    log = get_logger()
    log.info('data_loading')
    log.info('training_done', pipeline='P1', model='RandomForest')
    log.success('实验完成 | Experiment completed')
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# 导入配置
try:
    from config import LOGGER_CONFIG, BILINGUAL_MESSAGES, OUTPUT_DIR
except ImportError:
    # 默认配置
    LOGGER_CONFIG = {
        'language': 'bilingual',
        'level': 'INFO',
        'format': '[%(levelname)s] %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'file_output': True,
        'console_output': True,
    }
    BILINGUAL_MESSAGES = {}
    OUTPUT_DIR = Path('./output')


class BilingualFormatter(logging.Formatter):
    """双语日志格式化器"""
    
    def __init__(self, fmt=None, datefmt=None, language='bilingual'):
        super().__init__(fmt, datefmt)
        self.language = language
    
    def format(self, record):
        # 如果消息是消息键，则查找对应的双语内容
        if hasattr(record, 'msg_key') and record.msg_key in BILINGUAL_MESSAGES:
            msg_dict = BILINGUAL_MESSAGES[record.msg_key]
            if self.language == 'bilingual':
                record.msg = f"{msg_dict['zh']} | {msg_dict['en']}"
            elif self.language == 'chinese':
                record.msg = msg_dict['zh']
            elif self.language == 'english':
                record.msg = msg_dict['en']
        
        # 添加额外参数到消息
        if hasattr(record, 'extra_params') and record.extra_params:
            extra_str = ' | '.join([f"{k}={v}" for k, v in record.extra_params.items()])
            record.msg = f"{record.msg} [{extra_str}]"
        
        return super().format(record)


class BilingualLogger:
    """双语日志记录器"""
    
    def __init__(self, name='experiment', log_file=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOGGER_CONFIG.get('level', 'INFO')))
        
        # 清除已有处理器
        self.logger.handlers = []
        
        # 创建格式化器
        formatter = BilingualFormatter(
            fmt=LOGGER_CONFIG.get('format', '[%(levelname)s] %(message)s'),
            datefmt=LOGGER_CONFIG.get('date_format', '%Y-%m-%d %H:%M:%S'),
            language=LOGGER_CONFIG.get('language', 'bilingual')
        )
        
        # 控制台输出
        if LOGGER_CONFIG.get('console_output', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件输出
        if LOGGER_CONFIG.get('file_output', True) and log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _log(self, level, msg_key, extra_params=None):
        """内部日志方法"""
        extra = {'msg_key': msg_key, 'extra_params': extra_params or {}}
        self.logger.log(level, msg_key, extra=extra)
    
    def debug(self, msg_key, **kwargs):
        """调试日志"""
        self._log(logging.DEBUG, msg_key, kwargs)
    
    def info(self, msg_key, **kwargs):
        """信息日志"""
        self._log(logging.INFO, msg_key, kwargs)
    
    def warning(self, msg_key, **kwargs):
        """警告日志"""
        self._log(logging.WARNING, msg_key, kwargs)
    
    def error(self, msg_key, **kwargs):
        """错误日志"""
        self._log(logging.ERROR, msg_key, kwargs)
    
    def success(self, message):
        """成功日志（自定义消息）"""
        self.logger.info(f"[SUCCESS] {message}")
    
    def section(self, title_zh, title_en):
        """章节分隔日志"""
        self.logger.info("=" * 60)
        self.logger.info(f"{title_zh} | {title_en}")
        self.logger.info("=" * 60)
    
    def progress(self, current, total, msg_key='progress', **kwargs):
        """进度日志"""
        percentage = (current / total) * 100 if total > 0 else 0
        extra = {
            'current': current,
            'total': total,
            'percentage': f"{percentage:.1f}%",
            **kwargs
        }
        self._log(logging.INFO, msg_key, extra)


# 全局日志实例
_logger_instance = None

def get_logger(name='experiment', log_file=None):
    """获取日志记录器实例"""
    global _logger_instance
    if _logger_instance is None:
        if log_file is None:
            # 创建默认日志文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = OUTPUT_DIR / f"experiment_{timestamp}.log"
        _logger_instance = BilingualLogger(name, log_file)
    return _logger_instance


def reset_logger():
    """重置日志记录器（用于新实验）"""
    global _logger_instance
    _logger_instance = None


# 便捷函数
def log_info(msg_key, **kwargs):
    """快速记录信息日志"""
    get_logger().info(msg_key, **kwargs)

def log_success(message):
    """快速记录成功日志"""
    get_logger().success(message)

def log_warning(msg_key, **kwargs):
    """快速记录警告日志"""
    get_logger().warning(msg_key, **kwargs)

def log_error(msg_key, **kwargs):
    """快速记录错误日志"""
    get_logger().error(msg_key, **kwargs)


if __name__ == "__main__":
    # 测试日志系统
    log = get_logger()
    
    print("\n=== 测试双语日志系统 ===\n")
    
    log.section("实验开始", "Experiment Started")
    log.info('data_loading')
    log.info('data_loaded', samples_train=5000, samples_test=1000)
    log.info('preprocessing')
    log.info('preprocessing_done')
    log.info('training', pipeline='P1', model='RandomForest')
    log.info('training_done')
    log.success("所有实验完成 | All experiments completed")
    
    print("\n=== 测试完成 ===")
