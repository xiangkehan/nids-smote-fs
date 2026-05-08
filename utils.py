"""
通用工具函数模块

提供数据加载、评估指标计算、结果保存等通用功能
"""

import json
import pickle
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import config


def load_nsl_kdd_data(train_file=None, test_file=None):
    """
    加载NSL-KDD数据集
    
    参数:
        train_file: 训练集文件路径，默认使用config中的路径
        test_file: 测试集文件路径，默认使用config中的路径
    
    返回:
        train_df, test_df: 训练集和测试集的DataFrame
    """
    if train_file is None:
        train_file = config.TRAIN_FILE
    if test_file is None:
        test_file = config.TEST_FILE
    
    # NSL-KDD数据集的列名
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    # 加载数据
    train_df = pd.read_csv(train_file, names=column_names)
    test_df = pd.read_csv(test_file, names=column_names)
    
    return train_df, test_df


def map_attack_categories(df, attack_mapping=None):
    """
    将原始攻击类别映射为4大类
    
    参数:
        df: 包含'label'列的DataFrame
        attack_mapping: 攻击类别映射字典，默认使用config中的映射
    
    返回:
        添加'category'列的DataFrame
    """
    if attack_mapping is None:
        attack_mapping = config.ATTACK_CATEGORIES
    
    df = df.copy()
    df['category'] = df['label'].map(attack_mapping)
    
    # 处理未映射的类别（如果有的话，映射为'other'或保持原样）
    df['category'] = df['category'].fillna(df['label'])
    
    return df


def preprocess_features(df, categorical_features=None):
    """
    预处理特征：编码分类变量，分离特征和标签
    
    参数:
        df: 原始DataFrame
        categorical_features: 分类特征列名列表，默认使用config中的配置
    
    返回:
        X, y: 特征矩阵和标签向量
    """
    if categorical_features is None:
        # 使用列名而非索引
        categorical_features = ['protocol_type', 'service', 'flag']
    
    df = df.copy()
    
    # 删除不需要的列
    if 'difficulty' in df.columns:
        df = df.drop('difficulty', axis=1)
    
    # 分离特征和标签
    if 'category' in df.columns:
        y = df['category']
        X = df.drop(['label', 'category'], axis=1)
    else:
        y = df['label']
        X = df.drop('label', axis=1)
    
    # 对分类特征进行one-hot编码
    X = pd.get_dummies(X, columns=categorical_features, drop_first=False)
    
    return X, y


def encode_labels_for_xgboost(y):
    """
    为XGBoost编码标签（XGBoost需要数值型标签）
    
    参数:
        y: 原始标签（字符串）
    
    返回:
        y_encoded: 编码后的标签
        label_mapping: 标签映射字典
    """
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 创建映射字典
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    inverse_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    
    return y_encoded, label_mapping, inverse_mapping


def split_by_pipeline(X, y, pipeline_type='train_test'):
    """
    根据流水线类型划分数据
    
    参数:
        X: 特征矩阵
        y: 标签向量
        pipeline_type: 划分类型
    
    返回:
        划分后的数据
    """
    from sklearn.model_selection import train_test_split
    
    if pipeline_type == 'train_test':
        return train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)
    else:
        return X, y


def calculate_metrics(y_true, y_pred, average=None):
    """
    计算分类指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        average: 平均方式，None则返回每类指标
    
    返回:
        metrics_dict: 包含各项指标的字典
    """
    metrics = {}
    
    # 整体指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 每类指标
    if average is None:
        classes = np.unique(y_true)
        precision_per_class = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        
        for i, cls in enumerate(classes):
            metrics[f'precision_{cls}'] = precision_per_class[i]
            metrics[f'recall_{cls}'] = recall_per_class[i]
            metrics[f'f1_{cls}'] = f1_per_class[i]
    
    return metrics


def save_results(results, output_dir, filename='results.csv'):
    """
    保存实验结果到CSV文件（自动添加时间戳）
    
    参数:
        results: 结果字典或列表
        output_dir: 输出目录
        filename: 文件名
    
    返回:
        保存的文件路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if isinstance(results, dict):
        results = [results]
    
    # 为每个结果添加时间戳
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in results:
        if 'timestamp' not in r:
            r['timestamp'] = current_time
    
    df = pd.DataFrame(results)
    df.to_csv(output_path / filename, index=False)
    
    return output_path / filename


def save_model(model, output_dir, filename='model.pkl'):
    """
    保存训练好的模型
    
    参数:
        model: 训练好的模型
        output_dir: 输出目录
        filename: 文件名
    """
    if not config.OUTPUT['save_models']:
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / filename
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path


def save_checkpoint(checkpoint_data, output_dir, filename='checkpoint.json'):
    """
    保存检查点（断点继续机制）
    
    参数:
        checkpoint_data: 检查点数据字典
        output_dir: 输出目录
        filename: 文件名
    """
    if not config.CHECKPOINT_CONFIG['enabled']:
        return None
    
    checkpoint_dir = Path(output_dir) / config.CHECKPOINT_CONFIG['checkpoint_dir']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = checkpoint_dir / filename
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    return checkpoint_file


def load_checkpoint(output_dir, filename='checkpoint.json'):
    """
    加载检查点
    
    参数:
        output_dir: 输出目录
        filename: 文件名
    
    返回:
        检查点数据字典，如果不存在则返回None
    """
    checkpoint_dir = Path(output_dir) / config.CHECKPOINT_CONFIG['checkpoint_dir']
    checkpoint_file = checkpoint_dir / filename
    
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None


def format_time(seconds):
    """
    格式化时间（秒转分钟/小时）
    
    参数:
        seconds: 秒数
    
    返回:
        格式化后的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.2f}分钟"
    else:
        return f"{seconds/3600:.2f}小时"


def print_section_header(title_zh, title_en=""):
    """
    打印章节标题（双语，含时间戳）
    
    参数:
        title_zh: 中文标题
        title_en: 英文标题
    """
    current_time = datetime.now().strftime("%H:%M:%S")
    print("=" * 60)
    if title_en:
        print(f"[{current_time}] {title_zh} | {title_en}")
    else:
        print(f"[{current_time}] {title_zh}")
    print("=" * 60)


def create_output_subdir(base_dir, subdir_name):
    """
    创建输出子目录
    
    参数:
        base_dir: 基础输出目录
        subdir_name: 子目录名
    
    返回:
        子目录路径
    """
    subdir = Path(base_dir) / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def log_experiment_start(pipeline_name, model_name, **kwargs):
    """
    记录实验开始信息（含时间戳）
    
    参数:
        pipeline_name: 流水线名称
        model_name: 模型名称
        **kwargs: 其他参数
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[实验开始 | Experiment Started] [{current_time}]")
    print(f"流水线 | Pipeline: {pipeline_name}")
    print(f"模型 | Model: {model_name}")
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    print("-" * 60)


def log_experiment_end(pipeline_name, model_name, metrics, duration):
    """
    记录实验结束信息（含时间戳）
    
    参数:
        pipeline_name: 流水线名称
        model_name: 模型名称
        metrics: 评估指标字典
        duration: 运行时间（秒）
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[实验完成 | Experiment Completed] [{current_time}]")
    print(f"流水线 | Pipeline: {pipeline_name}")
    print(f"模型 | Model: {model_name}")
    print(f"运行时间 | Duration: {format_time(duration)}")
    print(f"准确率 | Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"宏平均F1 | Macro F1: {metrics.get('f1_macro', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    # 测试工具函数
    print("工具函数测试")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print(f"训练文件: {config.TRAIN_FILE}")
    print(f"测试文件: {config.TEST_FILE}")
