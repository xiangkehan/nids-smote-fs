"""
10_SMOTE变体对比实验 (SMOTE Variants Comparison)

功能:
对比不同过采样方法在入侵检测中的性能差异

对比方法:
1. RandomOverSampler (ROS) - 随机过采样基线
2. SMOTE - 基础版
3. BorderlineSMOTE - 边界版
4. ADASYN - 自适应版

输出:
- 各方法性能对比表
- 性能对比柱状图（中英双语）
- 每类召回率对比
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
)
from collections import Counter

import plot_config
import config
from utils import (
    calculate_metrics, save_results, save_model,
    log_experiment_start, log_experiment_end,
    create_output_subdir, print_section_header, encode_labels_for_xgboost
)
from baseline import get_model


def get_sampler(sampler_name, random_state=42):
    """
    获取过采样器实例
    
    参数:
        sampler_name: 过采样器名称 ('ROS', 'SMOTE', 'BorderlineSMOTE', 'ADASYN')
        random_state: 随机种子
    
    返回:
        sampler: 过采样器实例
    """
    samplers = {
        'ROS': RandomOverSampler(random_state=random_state),
        'SMOTE': SMOTE(random_state=random_state),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=random_state),
        'ADASYN': ADASYN(random_state=random_state)
    }
    
    if sampler_name not in samplers:
        raise ValueError(f"未知的过采样器: {sampler_name}. 可选: {list(samplers.keys())}")
    
    return samplers[sampler_name]


def apply_sampler(X_train, y_train, sampler_name, random_state=42):
    """
    应用过采样
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        sampler_name: 过采样器名称
        random_state: 随机种子
    
    返回:
        X_resampled, y_resampled, sampler_time
    """
    print(f"\n应用{sampler_name}过采样...")
    print(f"原始训练集分布:")
    for cls, count in sorted(Counter(y_train).items()):
        print(f"  {cls}: {count}")
    
    sampler = get_sampler(sampler_name, random_state)
    
    start_time = time.time()
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    sampler_time = time.time() - start_time
    
    print(f"\n{sampler_name}过采样完成！耗时: {sampler_time:.2f}秒")
    print(f"过采样后训练集分布:")
    for cls, count in sorted(Counter(y_resampled).items()):
        print(f"  {cls}: {count}")
    
    print(f"训练集从 {len(y_train)} 扩展到 {len(y_resampled)} 样本")
    
    return X_resampled, y_resampled, sampler_time


def run_sampler_comparison(X_train, y_train, X_test, y_test, output_dir, 
                            model_name='xgboost', pipeline_type='P2'):
    """
    运行过采样方法对比实验
    
    参数:
        model_name: 测试的模型名称
        pipeline_type: 'P2'(仅过采样) 或 'P4'(过采样→FS)
    """
    print_section_header(
        f"过采样方法对比实验 ({model_name}, {pipeline_type})",
        f"Sampler Comparison ({model_name}, {pipeline_type})"
    )
    
    sampler_dir = create_output_subdir(
        output_dir, 
        f'10_sampler_comparison_{model_name}_{pipeline_type}'
    )
    
    sampler_names = ['ROS', 'SMOTE', 'BorderlineSMOTE', 'ADASYN']
    results_list = []
    
    print(f"\n对比的过采样方法: {sampler_names}")
    print(f"测试模型: {model_name}")
    print(f"流水线类型: {pipeline_type}")
    
    for sampler_name in sampler_names:
        print(f"\n{'='*60}")
        print(f"测试 {sampler_name}")
        print(f"{'='*60}")
        
        try:
            # 应用过采样
            X_resampled, y_resampled, sampler_time = apply_sampler(
                X_train, y_train, sampler_name, config.RANDOM_STATE)
            
            # 如果使用P4流水线，需要特征选择
            if pipeline_type == 'P4':
                from feature_selection import wrapper_feature_selection
                X_resampled, X_test_fs, selected_features, _, fs_time = \
                    wrapper_feature_selection(X_resampled, y_resampled, X_test, k=30)
                print(f"特征选择完成，从{X_train.shape[1]}维降至30维")
            else:
                fs_time = 0
            
            # 训练模型
            model = get_model(model_name)
            
            start_time = time.time()
            if model_name == 'xgboost':
                y_resampled_model, label_mapping, inverse_mapping = encode_labels_for_xgboost(y_resampled)
                model.fit(X_resampled, y_resampled_model)
                y_pred_encoded = model.predict(X_test if pipeline_type == 'P2' else X_test_fs)
                y_pred = np.array([inverse_mapping[p] for p in y_pred_encoded])
            else:
                model.fit(X_resampled, y_resampled)
                y_pred = model.predict(X_test if pipeline_type == 'P2' else X_test_fs)
            
            train_time = time.time() - start_time
            
            # 计算指标
            metrics = calculate_metrics(y_test, y_pred, average=None)
            
            # 记录结果
            result = {
                'sampler': sampler_name,
                'model': model_name,
                'pipeline': pipeline_type,
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'recall_r2l': metrics.get('recall_r2l', 0),
                'recall_u2r': metrics.get('recall_u2r', 0),
                'recall_normal': metrics.get('recall_normal', 0),
                'recall_dos': metrics.get('recall_dos', 0),
                'recall_probe': metrics.get('recall_probe', 0),
                'train_time': train_time,
                'sampler_time': sampler_time,
                'fs_time': fs_time,
                'total_time': train_time + sampler_time + fs_time,
            }
            results_list.append(result)
            
            print(f"{sampler_name}: F1-macro={metrics['f1_macro']:.4f}, "
                  f"Accuracy={metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"错误: {sampler_name} 实验失败: {e}")
            # 记录失败结果
            result = {
                'sampler': sampler_name,
                'model': model_name,
                'pipeline': pipeline_type,
                'error': str(e)
            }
            results_list.append(result)
    
    # 保存结果
    results_df = pd.DataFrame(results_list)
    results_file = save_results(results_list, sampler_dir, 'sampler_comparison_results.csv')
    print(f"\n结果已保存: {results_file}")
    
    # 绘制对比图（双语）
    plot_sampler_comparison(results_df, sampler_dir, language='ch')
    plot_sampler_comparison(results_df, sampler_dir, language='en')
    
    # 绘制每类召回率对比（双语）
    plot_per_class_recall(results_df, sampler_dir, language='ch')
    plot_per_class_recall(results_df, sampler_dir, language='en')
    
    return results_list


def plot_sampler_comparison(results_df, output_dir, language='ch'):
    """
    绘制过采样方法性能对比图
    """
    if language == 'ch':
        labels = {
            'title': '过采样方法性能对比',
            'xlabel': '过采样方法',
            'ylabel': '性能指标',
            'f1_macro': 'F1-macro',
            'accuracy': '准确率',
        }
        filename = 'sampler_comparison_ch.png'
    else:
        labels = {
            'title': 'Sampler Performance Comparison',
            'xlabel': 'Sampler Method',
            'ylabel': 'Performance Metric',
            'f1_macro': 'F1-macro',
            'accuracy': 'Accuracy',
        }
        filename = 'sampler_comparison_en.png'
    
    # 过滤掉有错误的行
    valid_df = results_df[results_df['error'].isna() if 'error' in results_df.columns else True]
    
    if len(valid_df) == 0:
        print("警告: 没有有效的实验结果用于绘图")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    samplers = valid_df['sampler'].values
    x = np.arange(len(samplers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, valid_df['f1_macro'].values, width, 
                    label=labels['f1_macro'], color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, valid_df['accuracy'].values, width,
                    label=labels['accuracy'], color='coral', alpha=0.8)
    
    ax.set_xlabel(labels['xlabel'], fontsize=12)
    ax.set_ylabel(labels['ylabel'], fontsize=12)
    ax.set_title(labels['title'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(samplers, rotation=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"过采样方法对比图已保存: {output_path}")
    
    plt.close()


def plot_per_class_recall(results_df, output_dir, language='ch'):
    """
    绘制每类召回率对比图
    """
    if language == 'ch':
        labels = {
            'title': '每类召回率对比',
            'xlabel': '过采样方法',
            'ylabel': '召回率',
            'classes': ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
        }
        filename = 'per_class_recall_ch.png'
    else:
        labels = {
            'title': 'Per-Class Recall Comparison',
            'xlabel': 'Sampler Method',
            'ylabel': 'Recall',
            'classes': ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
        }
        filename = 'per_class_recall_en.png'
    
    valid_df = results_df[results_df['error'].isna() if 'error' in results_df.columns else True]
    
    if len(valid_df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    samplers = valid_df['sampler'].values
    classes = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    class_labels = labels['classes']
    
    x = np.arange(len(samplers))
    width = 0.15
    
    colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
    
    for i, (cls, cls_label, color) in enumerate(zip(classes, class_labels, colors)):
        col_name = f'recall_{cls}'
        if col_name in valid_df.columns:
            offset = width * (i - 2)
            ax.bar(x + offset, valid_df[col_name].values, width,
                   label=cls_label, color=color, alpha=0.8)
    
    ax.set_xlabel(labels['xlabel'], fontsize=12)
    ax.set_ylabel(labels['ylabel'], fontsize=12)
    ax.set_title(labels['title'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(samplers, rotation=15)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"每类召回率对比图已保存: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    print_section_header(
        "阶段10：SMOTE变体对比实验",
        "Stage 10: SMOTE Variants Comparison"
    )
    
    # 创建输出目录
    output_dir = config.create_output_dir()
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    print("\n加载数据...")
    from utils import load_nsl_kdd_data, map_attack_categories, preprocess_features
    
    train_df, test_df = load_nsl_kdd_data()
    
    # 调试模式采样
    if config.DEBUG_MODE:
        print(f"[调试模式] 采样 {config.DEBUG_TRAIN_SIZE} 训练样本")
        train_df = train_df.sample(n=min(config.DEBUG_TRAIN_SIZE, len(train_df)),
                                   random_state=config.RANDOM_STATE)
        test_df = test_df.sample(n=min(config.DEBUG_TEST_SIZE, len(test_df)),
                                random_state=config.RANDOM_STATE)
    
    # 预处理
    train_df = map_attack_categories(train_df)
    test_df = map_attack_categories(test_df)
    
    X_train, y_train = preprocess_features(train_df)
    X_test, y_test = preprocess_features(test_df)
    
    # 对齐特征
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    
    # 1. P2流水线对比
    print("\n" + "="*60)
    print("1. P2流水线：仅过采样")
    print("="*60)
    p2_results = run_sampler_comparison(
        X_train, y_train, X_test, y_test, 
        output_dir, model_name='xgboost', pipeline_type='P2'
    )
    
    # 2. P4流水线对比
    print("\n" + "="*60)
    print("2. P4流水线：过采样→特征选择")
    print("="*60)
    p4_results = run_sampler_comparison(
        X_train, y_train, X_test, y_test,
        output_dir, model_name='xgboost', pipeline_type='P4'
    )
    
    print_section_header(
        "SMOTE变体对比实验完成",
        "SMOTE Variants Comparison Completed"
    )
    
    return p2_results, p4_results


if __name__ == "__main__":
    main()
