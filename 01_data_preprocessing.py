"""
01_数据预处理与探索性分析

功能:
1. 加载NSL-KDD数据集
2. 分析类别分布和不平衡程度
3. 编码分类特征
4. 保存预处理后的数据和统计信息

输出:
- 预处理后的训练集和测试集
- 类别分布可视化
- 数据统计信息
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 导入可视化配置（设置中文字体）
import plot_config

import config
from utils import (
    load_nsl_kdd_data, map_attack_categories, preprocess_features,
    save_results, create_output_subdir, print_section_header
)


def analyze_class_distribution(y, title="类别分布"):
    """
    分析并可视化类别分布
    
    参数:
        y: 标签向量
        title: 图表标题
    
    返回:
        distribution_df: 类别分布DataFrame
    """
    counter = Counter(y)
    total = sum(counter.values())
    
    # 创建分布表
    distribution = []
    for cls, count in sorted(counter.items()):
        percentage = (count / total) * 100
        distribution.append({
            '类别 | Class': cls,
            '样本数 | Count': count,
            '占比 | Percentage': f"{percentage:.2f}%",
            '占比数值 | Percentage_Value': percentage
        })
    
    distribution_df = pd.DataFrame(distribution)
    
    # 计算不平衡比率
    max_count = max(counter.values())
    min_count = min(counter.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\n类别分布统计 | Class Distribution Statistics:")
    print(f"总样本数 | Total Samples: {total}")
    print(f"类别数 | Number of Classes: {len(counter)}")
    print(f"最大类样本数 | Max Class Count: {max_count}")
    print(f"最小类样本数 | Min Class Count: {min_count}")
    print(f"不平衡比率 | Imbalance Ratio: {imbalance_ratio:.2f}")
    print("\n详细分布:")
    print(distribution_df.to_string(index=False))
    
    return distribution_df, imbalance_ratio


def plot_class_distribution(y, output_dir, filename='class_distribution.png'):
    """
    绘制类别分布图（纯中文）
    
    参数:
        y: 标签向量
        output_dir: 输出目录
        filename: 文件名
    """
    counter = Counter(y)
    classes = list(counter.keys())
    counts = list(counter.values())
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 柱状图
    bars = ax1.bar(classes, counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('攻击类别', fontsize=12)
    ax1.set_ylabel('样本数量', fontsize=12)
    ax1.set_title('训练集类别分布', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # 饼图
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    wedges, texts, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
    ax2.set_title('训练集类别占比', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'], 
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"\n类别分布图已保存: {output_path}")
    
    plt.close()


def plot_class_imbalance_ratio(y, output_dir, filename='imbalance_ratio.png'):
    """
    绘制不平衡比率图（纯中文）
    
    参数:
        y: 标签向量
        output_dir: 输出目录
        filename: 文件名
    """
    counter = Counter(y)
    classes = list(counter.keys())
    counts = np.array(list(counter.values()))
    
    # 计算每个类别相对于最大类的比率
    max_count = counts.max()
    ratios = max_count / counts
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red' if r > 10 else 'orange' if r > 5 else 'green' for r in ratios]
    bars = ax.bar(classes, ratios, color=colors, alpha=0.7)
    
    ax.set_xlabel('攻击类别', fontsize=12)
    ax.set_ylabel('不平衡比率（对数尺度）', fontsize=12)
    ax.set_title('各类别不平衡程度分析', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='严重不平衡阈值')
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='中度不平衡阈值')
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}倍',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"不平衡比率图已保存: {output_path}")
    
    plt.close()


def preprocess_and_save_data(train_df, test_df, output_dir):
    """
    预处理数据并保存
    
    参数:
        train_df: 训练集DataFrame
        test_df: 测试集DataFrame
        output_dir: 输出目录
    
    返回:
        X_train, y_train, X_test, y_test: 预处理后的数据
    """
    print_section_header("数据预处理 | Data Preprocessing", "")
    
    # 1. 攻击类别映射
    print("\n映射攻击类别 | Mapping attack categories...")
    train_df = map_attack_categories(train_df)
    test_df = map_attack_categories(test_df)
    
    # 2. 特征预处理
    print("预处理特征 | Preprocessing features...")
    X_train, y_train = preprocess_features(train_df)
    X_test, y_test = preprocess_features(test_df)
    
    # 3. 对齐训练集和测试集的特征
    print("对齐特征 | Aligning features...")
    # 获取训练集和测试集的共同特征
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"特征数量 | Number of features: {len(common_features)}")
    print(f"训练样本数 | Training samples: {len(X_train)}")
    print(f"测试样本数 | Test samples: {len(X_test)}")
    
    # 4. 保存预处理后的数据
    print("保存预处理数据 | Saving preprocessed data...")
    
    # 保存特征名
    feature_names_file = Path(output_dir) / 'feature_names.txt'
    with open(feature_names_file, 'w', encoding='utf-8') as f:
        for i, feat in enumerate(common_features):
            f.write(f"{i}: {feat}\n")
    print(f"特征名已保存 | Feature names saved: {feature_names_file}")
    
    # 保存为numpy数组
    np.savez(Path(output_dir) / 'preprocessed_data.npz',
             X_train=X_train.values, y_train=y_train.values,
             X_test=X_test.values, y_test=y_test.values)
    print(f"预处理数据已保存 | Preprocessed data saved")
    
    # 保存为CSV（方便查看）
    train_processed = pd.concat([X_train, y_train.rename('label')], axis=1)
    test_processed = pd.concat([X_test, y_test.rename('label')], axis=1)
    
    train_processed.to_csv(Path(output_dir) / 'train_processed.csv', index=False)
    test_processed.to_csv(Path(output_dir) / 'test_processed.csv', index=False)
    print(f"CSV格式数据已保存 | CSV data saved")
    
    return X_train, y_train, X_test, y_test


def main():
    """
    主函数：执行数据预处理和探索性分析
    """
    print_section_header(
        "阶段1：数据预处理与探索性分析",
        "Stage 1: Data Preprocessing and Exploratory Analysis"
    )
    
    # 创建输出目录
    output_dir = config.create_output_dir()
    data_exploration_dir = create_output_subdir(output_dir, '01_data_exploration')
    
    print(f"\n输出目录 | Output directory: {output_dir}")
    
    # 1. 加载数据
    print_section_header("加载数据 | Loading Data", "")
    print(f"训练集 | Training set: {config.TRAIN_FILE}")
    print(f"测试集 | Test set: {config.TEST_FILE}")
    
    train_df, test_df = load_nsl_kdd_data()
    
    print(f"训练集大小 | Training set size: {len(train_df)}")
    print(f"测试集大小 | Test set size: {len(test_df)}")
    
    # 2. 调试模式：采样
    if config.DEBUG_MODE:
        print(f"\n[调试模式 | DEBUG MODE]")
        print(f"采样训练集 | Sampling training set: {config.DEBUG_TRAIN_SIZE} samples")
        print(f"采样测试集 | Sampling test set: {config.DEBUG_TEST_SIZE} samples")
        
        train_df = train_df.sample(n=min(config.DEBUG_TRAIN_SIZE, len(train_df)), 
                                   random_state=config.RANDOM_STATE)
        test_df = test_df.sample(n=min(config.DEBUG_TEST_SIZE, len(test_df)),
                                random_state=config.RANDOM_STATE)
    
    # 3. 映射攻击类别
    print_section_header("攻击类别映射 | Attack Category Mapping", "")
    train_df = map_attack_categories(train_df)
    test_df = map_attack_categories(test_df)
    
    # 4. 分析训练集类别分布
    print_section_header("训练集类别分布分析 | Training Set Class Distribution", "")
    train_dist, train_imbalance = analyze_class_distribution(
        train_df['category'], "训练集类别分布 | Training Set Class Distribution"
    )
    
    # 5. 分析测试集类别分布
    print_section_header("测试集类别分布分析 | Test Set Class Distribution", "")
    test_dist, test_imbalance = analyze_class_distribution(
        test_df['category'], "测试集类别分布 | Test Set Class Distribution"
    )
    
    # 6. 可视化
    print_section_header("生成可视化 | Generating Visualizations", "")
    plot_class_distribution(train_df['category'], data_exploration_dir, 
                           'train_class_distribution.png')
    plot_class_distribution(test_df['category'], data_exploration_dir,
                           'test_class_distribution.png')
    plot_class_imbalance_ratio(train_df['category'], data_exploration_dir,
                              'train_imbalance_ratio.png')
    
    # 7. 保存统计信息
    stats = {
        'dataset': 'NSL-KDD',
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'train_imbalance_ratio': train_imbalance,
        'test_imbalance_ratio': test_imbalance,
        'num_features': len(train_df.columns) - 3,  # 减去label, category, difficulty
        'num_classes': len(train_df['category'].unique()),
        'debug_mode': config.DEBUG_MODE
    }
    
    save_results(stats, data_exploration_dir, 'data_statistics.csv')
    
    # 8. 预处理数据
    print_section_header("预处理数据 | Preprocessing Data", "")
    X_train, y_train, X_test, y_test = preprocess_and_save_data(
        train_df, test_df, data_exploration_dir
    )
    
    print_section_header(
        "数据预处理完成 | Data Preprocessing Completed",
        ""
    )
    print(f"所有结果已保存至 | All results saved to: {data_exploration_dir}")
    
    return X_train, y_train, X_test, y_test, output_dir


if __name__ == "__main__":
    main()
