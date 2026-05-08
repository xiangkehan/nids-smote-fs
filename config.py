"""
面向类别不平衡入侵检测的SMOTE与特征选择联合流水线：配置驱动文件

使用说明:
1. 修改本文件中的配置参数
2. 运行各阶段实验脚本
3. 脚本会自动读取本配置

调试模式:
- 设置 DEBUG_MODE = True 可进行小规模快速测试
- 设置 DEBUG_SAMPLE_SIZE 控制测试样本数
"""

import os
from pathlib import Path

# ==================== 基础路径配置 ====================
# 项目根目录（自动计算）
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

# 输出目录（按时间戳自动创建子目录）
OUTPUT_DIR = PROJECT_ROOT / "output"

# 代码目录
CODE_DIR = PROJECT_ROOT / "code"

# ==================== 调试模式配置 ====================
# 开启调试模式：使用小批量数据快速测试
DEBUG_MODE = False

# 调试模式下的样本数量（仅用于快速验证代码逻辑）
DEBUG_TRAIN_SIZE = 5000   # 训练集采样数量
DEBUG_TEST_SIZE = 1000    # 测试集采样数量

# ==================== 重复实验配置 ====================
# 用于统计显著性检验
REPEAT_EXPERIMENTS = {
    'enabled': True,           # 是否启用重复实验
    'n_repeats': 5,            # 重复次数
    'random_states': [42, 123, 456, 789, 2024],  # 随机种子列表
}

# ==================== 数据集配置 ====================
# 支持的数据集
DATASETS = {
    'nsl_kdd': {
        'train_file': DATA_DIR / "KDDTrain+.txt",
        'test_file': DATA_DIR / "KDDTest+.txt",
        'enabled': True,
    },
    'unsw_nb15': {
        'train_file': DATA_DIR / "UNSW_NB15_training-set.csv",
        'test_file': DATA_DIR / "UNSW_NB15_testing-set.csv",
        'enabled': True,
    }
}

# ==================== 数据预处理配置 ====================
# 分类特征列索引（NSL-KDD中的3个分类特征）
CATEGORICAL_FEATURES = [1, 2, 3]  # protocol_type, service, flag

# 目标标签列名
TARGET_COLUMN = "label"

# 需要映射的4大类攻击类别
ATTACK_CATEGORIES = {
    'normal': 'normal',
    'neptune': 'dos', 'smurf': 'dos', 'back': 'dos', 'teardrop': 'dos', 
    'pod': 'dos', 'land': 'dos', 'apache2': 'dos', 'processtable': 'dos', 
    'mailbomb': 'dos', 'udpstorm': 'dos',
    'ipsweep': 'probe', 'portsweep': 'probe', 'nmap': 'probe', 
    'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    'guess_passwd': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l', 
    'imap': 'r2l', 'ftp_write': 'r2l', 'phf': 'r2l', 'multihop': 'r2l', 
    'spy': 'r2l', 'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l', 
    'snmpguess': 'r2l', 'worm': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 
    'rootkit': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 
    'xterm': 'u2r'
}

# 5分类标签
CLASS_LABELS = ['normal', 'dos', 'probe', 'r2l', 'u2r']

# ==================== 实验流水线配置 ====================
# 可以单独启用/禁用某个流水线进行测试
PIPELINES = {
    'P1_baseline': True,           # 基线模型
    'P2_smote_only': True,          # 仅SMOTE
    'P3_feature_selection_only': True,  # 仅特征选择
    'P4_smote_then_fs': True,       # SMOTE → 特征选择
    'P5_fs_then_smote': True,       # 特征选择 → SMOTE（核心假设）
}

# ==================== 机器学习模型配置 ====================
# 可以单独启用/禁用某个模型
MODELS = {
    'decision_tree': {
        'enabled': True,
        'params': {
            'random_state': 42,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
    },
    'random_forest': {
        'enabled': True,
        'params': {
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'n_jobs': -1
        }
    },
    'xgboost': {
        'enabled': True,
        'params': {
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1
        }
    }
}

# ==================== 特征选择配置 ====================
# 可以单独启用/禁用某种特征选择方法
FEATURE_SELECTION = {
    'filter_method': {
        'enabled': True,
        'method': 'mutual_info',  # 信息增益
        'thresholds': [10, 20, 30]  # Top-k特征数量
    },
    'wrapper_method': {
        'enabled': True,
        'estimator': 'random_forest',  # RFE使用的基学习器
        'thresholds': [10, 20, 30]  # Top-k特征数量
    }
}

# ==================== SMOTE配置 ====================
SMOTE_CONFIG = {
    'method': 'SMOTE',  # 可选：SMOTE, BorderlineSMOTE, ADASYN
    'random_state': 42,
    'k_neighbors': 5,   # 近邻数量
    # 采样策略：可以设置为'auto'（平衡所有类）或字典指定各类别采样倍数
    'sampling_strategy': 'auto'
}

# ==================== 评估指标配置 ====================
# 需要计算的指标
METRICS = [
    'accuracy',
    'precision_macro', 'precision_weighted',
    'recall_macro', 'recall_weighted',
    'f1_macro', 'f1_weighted'
]

# 需要输出每类详细指标的标签
PER_CLASS_METRICS = True

# ==================== 可视化配置 ====================
VISUALIZATION = {
    'dpi': 300,              # 图片分辨率
    'figsize': (12, 8),      # 默认图尺寸
    'style': 'seaborn-v0_8-whitegrid',  # matplotlib样式
    'format': 'png',         # 图片格式
    'bbox_inches': 'tight',  # 去除白边
    'font_family': 'SimHei',  # 中文字体
    'font_size': 12,         # 字体大小
}

# ==================== 输出配置 ====================
OUTPUT = {
    'save_models': True,           # 是否保存训练好的模型
    'save_predictions': True,      # 是否保存预测结果
    'save_feature_importance': True,  # 是否保存特征重要性
    'save_confusion_matrix': True,    # 是否保存混淆矩阵数据
    'verbose': True,               # 是否打印详细日志
    'log_level': 'INFO'            # 日志级别
}

# ==================== 双语日志配置 ====================
LOGGER_CONFIG = {
    'language': 'bilingual',  # 可选: 'chinese', 'english', 'bilingual'
    'level': 'INFO',          # 日志级别: DEBUG, INFO, WARNING, ERROR
    'format': '[%(levelname)s] %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'file_output': True,      # 是否输出到文件
    'console_output': True,   # 是否输出到控制台
}

# 预定义的双语日志消息模板
BILINGUAL_MESSAGES = {
    'data_loading': {'zh': '数据加载中...', 'en': 'Loading data...'},
    'data_loaded': {'zh': '数据加载完成', 'en': 'Data loaded successfully'},
    'preprocessing': {'zh': '数据预处理中...', 'en': 'Preprocessing data...'},
    'preprocessing_done': {'zh': '数据预处理完成', 'en': 'Preprocessing completed'},
    'training': {'zh': '模型训练中...', 'en': 'Training model...'},
    'training_done': {'zh': '模型训练完成', 'en': 'Model training completed'},
    'evaluating': {'zh': '模型评估中...', 'en': 'Evaluating model...'},
    'evaluation_done': {'zh': '模型评估完成', 'en': 'Evaluation completed'},
    'saving_results': {'zh': '保存结果中...', 'en': 'Saving results...'},
    'results_saved': {'zh': '结果保存完成', 'en': 'Results saved'},
    'checkpoint_saved': {'zh': '检查点已保存', 'en': 'Checkpoint saved'},
    'checkpoint_loaded': {'zh': '检查点已加载，继续实验', 'en': 'Checkpoint loaded, resuming experiment'},
    'experiment_completed': {'zh': '实验完成', 'en': 'Experiment completed'},
    'skipping_completed': {'zh': '跳过已完成实验', 'en': 'Skipping completed experiment'},
    'error_occurred': {'zh': '发生错误', 'en': 'Error occurred'},
}

# ==================== 断点继续配置 ====================
CHECKPOINT_CONFIG = {
    'enabled': True,           # 是否启用断点继续
    'auto_save': True,         # 每完成一组实验自动保存
    'save_interval': 1,        # 每隔多少组实验保存一次
    'checkpoint_dir': 'checkpoints',  # 检查点目录名
    'file_name': 'experiment_state.json',  # 状态文件名
    'completed_file': 'completed_experiments.csv',  # 已完成实验记录
}

# ==================== 随机种子配置 ====================
# 保证实验可复现
RANDOM_STATE = 42

# ==================== 并行计算配置 ====================
PARALLEL = {
    'n_jobs': -1,  # -1表示使用所有CPU核心
    'backend': 'loky'
}

# ==================== 论文图表配置 ====================
PAPER_FIGURES = {
    'font_size': 12,
    'font_family': 'SimHei',  # 中文字体
    'color_palette': 'Set2',   # 颜色主题
    'figure_format': 'pdf'     # 论文图表格式（高分辨率）
}

# ==================== 辅助函数 ====================
def get_active_models():
    """获取启用的模型列表"""
    return {name: config for name, config in MODELS.items() if config['enabled']}

def get_active_pipelines():
    """获取启用的流水线列表"""
    return {name: enabled for name, enabled in PIPELINES.items() if enabled}

def get_active_fs_methods():
    """获取启用的特征选择方法"""
    methods = {}
    if FEATURE_SELECTION['filter_method']['enabled']:
        methods['filter'] = FEATURE_SELECTION['filter_method']
    if FEATURE_SELECTION['wrapper_method']['enabled']:
        methods['wrapper'] = FEATURE_SELECTION['wrapper_method']
    return methods

def create_output_dir():
    """创建带时间戳的输出目录"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = OUTPUT_DIR / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def print_config():
    """打印当前配置信息（双语）"""
    print("=" * 60)
    print("当前实验配置 | Current Experiment Configuration")
    print("=" * 60)
    print(f"调试模式 | Debug Mode: {DEBUG_MODE}")
    if DEBUG_MODE:
        print(f"  训练样本数 | Training Samples: {DEBUG_TRAIN_SIZE}")
        print(f"  测试样本数 | Test Samples: {DEBUG_TEST_SIZE}")
    print(f"\n启用的流水线 | Active Pipelines: {list(get_active_pipelines().keys())}")
    print(f"启用的模型 | Active Models: {list(get_active_models().keys())}")
    print(f"启用的特征选择 | Active Feature Selection: {list(get_active_fs_methods().keys())}")
    print(f"SMOTE方法 | SMOTE Method: {SMOTE_CONFIG['method']}")
    print(f"双语日志 | Bilingual Logging: {LOGGER_CONFIG['language']}")
    print(f"断点继续 | Checkpoint Resume: {CHECKPOINT_CONFIG['enabled']}")
    print("=" * 60)

# 如果直接运行此文件，打印配置
if __name__ == "__main__":
    print_config()
