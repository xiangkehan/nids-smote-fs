"""
03_SMOTE过采样实验 (P2-SMOTE Only)

功能:
先对训练数据进行SMOTE过采样，然后在平衡数据上训练模型

流水线:
原始数据 → SMOTE过采样 → 模型训练 → 评估

输出:
- 过采样后的类别分布
- 各模型性能指标
- 与基线的对比
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
from collections import Counter
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# 导入可视化配置（设置中文字体）
import plot_config

import config
from utils import (
    calculate_metrics, save_results, save_model,
    log_experiment_start, log_experiment_end,
    create_output_subdir, print_section_header,
    encode_labels_for_xgboost
)


def get_model(model_name):
    """获取模型实例"""
    models_config = config.get_active_models()
    
    if model_name not in models_config:
        raise ValueError(f"未知模型: {model_name}")
    
    model_config = models_config[model_name]
    params = model_config['params']
    
    if model_name == 'decision_tree':
        return DecisionTreeClassifier(**params)
    elif model_name == 'random_forest':
        return RandomForestClassifier(**params)
    elif model_name == 'xgboost':
        return xgb.XGBClassifier(**params)
    else:
        raise ValueError(f"未实现的模型: {model_name}")


def apply_smote(X_train, y_train, smote_config=None):
    """
    应用SMOTE过采样（自动调整k_neighbors）
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        smote_config: SMOTE配置
    
    返回:
        X_resampled, y_resampled: 过采样后的数据
    """
    if smote_config is None:
        smote_config = config.SMOTE_CONFIG
    
    print(f"\n应用SMOTE过采样...")
    print(f"原始训练集分布:")
    for cls, count in sorted(Counter(y_train).items()):
        print(f"  {cls}: {count}")
    
    # 自动调整k_neighbors：不能超过最小类样本数-1
    min_class_count = min(Counter(y_train).values())
    k_neighbors = min(smote_config['k_neighbors'], min_class_count - 1)
    k_neighbors = max(k_neighbors, 1)  # 至少为1
    
    if k_neighbors < smote_config['k_neighbors']:
        print(f"警告: 最小类只有{min_class_count}个样本，自动调整k_neighbors从{smote_config['k_neighbors']}到{k_neighbors}")
    
    # 创建SMOTE实例
    smote = SMOTE(
        k_neighbors=k_neighbors,
        sampling_strategy=smote_config['sampling_strategy'],
        random_state=smote_config['random_state']
    )
    
    # 应用SMOTE
    start_time = time.time()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    smote_time = time.time() - start_time
    
    print(f"\nSMOTE过采样完成！耗时: {smote_time:.2f}秒")
    print(f"过采样后训练集分布:")
    for cls, count in sorted(Counter(y_resampled).items()):
        print(f"  {cls}: {count}")
    
    print(f"训练集从 {len(y_train)} 扩展到 {len(y_resampled)} 样本")
    
    return X_resampled, y_resampled, smote_time


def plot_resampled_distribution(y_before, y_after, output_dir, filename='smote_distribution.png'):
    """
    绘制SMOTE前后的类别分布对比图（纯中文）
    """
    counter_before = Counter(y_before)
    counter_after = Counter(y_after)
    
    classes = sorted(counter_before.keys())
    counts_before = [counter_before[c] for c in classes]
    counts_after = [counter_after[c] for c in classes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 过采样前
    bars1 = ax1.bar(classes, counts_before, color='lightcoral', alpha=0.7)
    ax1.set_xlabel('攻击类别', fontsize=12)
    ax1.set_ylabel('样本数量', fontsize=12)
    ax1.set_title('SMOTE过采样前', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 过采样后
    bars2 = ax2.bar(classes, counts_after, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('攻击类别', fontsize=12)
    ax2.set_ylabel('样本数量', fontsize=12)
    ax2.set_title('SMOTE过采样后', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"过采样对比图已保存: {output_path}")
    
    plt.close()


def run_smote_only_experiment(X_train, y_train, X_test, y_test, output_dir):
    """
    运行SMOTE实验
    
    流水线: 原始数据 → SMOTE过采样 → 模型训练 → 评估
    """
    print_section_header(
        "P2-SMOTE过采样实验",
        "在SMOTE平衡数据上训练模型"
    )
    
    smote_dir = create_output_subdir(output_dir, '03_smote_only')
    results_list = []
    
    # 1. 应用SMOTE
    X_resampled, y_resampled, smote_time = apply_smote(X_train, y_train)
    
    # 2. 可视化SMOTE效果
    plot_resampled_distribution(y_train, y_resampled, smote_dir)
    
    # 3. 在过采样后的数据上训练模型
    active_models = config.get_active_models()
    total_models = len(active_models)
    
    print(f"\n将训练 {total_models} 个模型")
    
    for idx, (model_name, model_config) in enumerate(active_models.items(), 1):
        print(f"\n[{idx}/{total_models}] 训练模型: {model_name}")
        
        log_experiment_start(
            pipeline_name='P2_smote_only',
            model_name=model_name,
            train_samples_original=len(y_train),
            train_samples_resampled=len(y_resampled),
            test_samples=len(y_test)
        )
        
        # 获取模型
        model = get_model(model_name)
        
        # 训练
        start_time = time.time()
        print(f"  训练中...")
        
        # XGBoost需要数值型标签
        if model_name == 'xgboost':
            y_resampled_model, label_mapping, inverse_mapping = encode_labels_for_xgboost(y_resampled)
            model.fit(X_resampled, y_resampled_model)
            
            # 预测
            print(f"  预测中...")
            y_pred_encoded = model.predict(X_test)
            # 转回原始标签
            y_pred = np.array([inverse_mapping[p] for p in y_pred_encoded])
        else:
            model.fit(X_resampled, y_resampled)
            
            # 预测
            print(f"  预测中...")
            y_pred = model.predict(X_test)
        
        train_time = time.time() - start_time
        
        # 计算指标
        print(f"  计算指标...")
        metrics = calculate_metrics(y_test, y_pred, average=None)
        
        # 添加实验信息
        metrics['pipeline'] = 'P2_smote_only'
        metrics['model'] = model_name
        metrics['train_time'] = train_time
        metrics['smote_time'] = smote_time
        metrics['train_samples_original'] = len(y_train)
        metrics['train_samples_resampled'] = len(y_resampled)
        metrics['test_samples'] = len(y_test)
        metrics['num_features'] = X_train.shape[1]
        
        # 保存结果
        results_list.append(metrics)
        
        # 保存混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm,
                            index=sorted(np.unique(y_test)),
                            columns=sorted(np.unique(y_test)))
        cm_df.to_csv(smote_dir / f'confusion_matrix_{model_name}.csv')
        
        # 保存模型
        if config.OUTPUT['save_models']:
            save_model(model, smote_dir, f'model_{model_name}.pkl')
        
        # 记录实验结束
        log_experiment_end('P2_smote_only', model_name, metrics, train_time)
    
    # 保存所有结果
    results_df = pd.DataFrame(results_list)
    results_file = save_results(results_list, smote_dir, 'results.csv')
    print(f"\n所有结果已保存: {results_file}")
    
    # 打印汇总表
    print("\n" + "=" * 60)
    print("SMOTE实验结果汇总")
    print("=" * 60)
    summary = results_df[['model', 'accuracy', 'f1_macro', 'f1_weighted', 'train_time']]
    print(summary.to_string(index=False))
    
    return results_list


def main():
    """主函数"""
    print_section_header(
        "阶段3：SMOTE过采样实验",
        ""
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
    
    # 运行SMOTE实验
    results = run_smote_only_experiment(X_train, y_train, X_test, y_test, output_dir)
    
    print_section_header(
        "SMOTE实验完成",
        ""
    )
    
    return results


if __name__ == "__main__":
    main()
