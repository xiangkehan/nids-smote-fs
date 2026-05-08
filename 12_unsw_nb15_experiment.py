"""
12_UNSW-NB15数据集实验 (UNSW-NB15 Dataset Experiment)

功能:
在UNSW-NB15数据集上复现核心实验配置

数据预处理:
- UNSW-NB15与NSL-KDD结构不同（49维原始特征）
- 攻击类别映射：Normal + 9种攻击 → 合并为DoS/Probe/R2L/U2R
- 特征编码：One-Hot编码分类特征

实验配置:
- P1: 基线
- P2: 仅SMOTE
- P4: SMOTE→FS (Wrapper, K=30)
- P5: FS→SMOTE (Wrapper, K=30)

输出:
- 各配置性能对比
- 与NSL-KDD结果对比图（中英双语）
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
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from collections import Counter

import plot_config
import config
from utils import (
    calculate_metrics, save_results, save_model,
    log_experiment_start, log_experiment_end,
    create_output_subdir, print_section_header, encode_labels_for_xgboost
)
from baseline import get_model
from smote_only import apply_smote
from feature_selection import filter_feature_selection, wrapper_feature_selection


# UNSW-NB15攻击类别映射
UNSW_ATTACK_MAPPING = {
    'Normal': 'normal',
    'Fuzzers': 'dos',
    'Analysis': 'probe',
    'Backdoors': 'r2l',
    'DoS': 'dos',
    'Exploits': 'u2r',
    'Generic': 'probe',
    'Reconnaissance': 'probe',
    'Shellcode': 'u2r',
    'Worms': 'u2r',
}


def load_unsw_data(train_file=None, test_file=None):
    """
    加载UNSW-NB15数据集
    
    返回:
        train_df, test_df: 训练集和测试集的DataFrame
    """
    if train_file is None:
        train_file = config.DATASETS['unsw_nb15']['train_file']
    if test_file is None:
        test_file = config.DATASETS['unsw_nb15']['test_file']
    
    print(f"\n加载UNSW-NB15数据集...")
    print(f"训练集: {train_file}")
    print(f"测试集: {test_file}")
    
    # 加载数据（UNSW是CSV格式，有表头）
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    return train_df, test_df


def preprocess_unsw_data(train_df, test_df):
    """
    预处理UNSW-NB15数据
    
    步骤:
    1. 攻击类别映射
    2. 删除无用列（id, attack_cat等）
    3. One-Hot编码分类特征
    4. 分离X和y
    
    返回:
        X_train, y_train, X_test, y_test
    """
    print("\n预处理UNSW-NB15数据...")
    
    # 1. 攻击类别映射
    # UNSW有 'label' 列（0=Normal, 1=Attack）和 'attack_cat' 列
    if 'attack_cat' in train_df.columns:
        # 使用attack_cat进行映射
        train_df['category'] = train_df['attack_cat'].map(UNSW_ATTACK_MAPPING)
        test_df['category'] = test_df['attack_cat'].map(UNSW_ATTACK_MAPPING)
        
        # 处理未映射的类别
        train_df['category'] = train_df['category'].fillna('other')
        test_df['category'] = test_df['category'].fillna('other')
    else:
        # 如果没有attack_cat，使用label列
        train_df['category'] = train_df['label'].apply(lambda x: 'normal' if x == 0 else 'attack')
        test_df['category'] = test_df['label'].apply(lambda x: 'normal' if x == 0 else 'attack')
    
    # 2. 删除无用列
    drop_cols = ['id', 'attack_cat', 'label']
    drop_cols = [col for col in drop_cols if col in train_df.columns]
    
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)
    
    # 3. 识别分类特征
    categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
    # 排除category列
    categorical_features = [col for col in categorical_features if col != 'category']
    
    print(f"分类特征: {categorical_features}")
    
    # 4. 分离X和y
    y_train = train_df['category']
    y_test = test_df['category']
    X_train = train_df.drop('category', axis=1)
    X_test = test_df.drop('category', axis=1)
    
    # 5. One-Hot编码
    # 合并训练和测试以确保编码一致
    combined = pd.concat([X_train, X_test], axis=0)
    combined_encoded = pd.get_dummies(combined, columns=categorical_features, drop_first=False)
    
    X_train = combined_encoded.iloc[:len(X_train)]
    X_test = combined_encoded.iloc[len(X_train):]
    
    # 对齐特征
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"预处理后训练集: {X_train.shape}")
    print(f"预处理后测试集: {X_test.shape}")
    print(f"\n训练集类别分布:")
    for cls, count in sorted(Counter(y_train).items()):
        print(f"  {cls}: {count}")
    
    return X_train, y_train, X_test, y_test


def run_unsw_experiment(X_train, y_train, X_test, y_test, output_dir):
    """
    运行UNSW-NB15核心实验配置
    
    配置:
    - P1: 基线 (XGBoost)
    - P2: 仅SMOTE (XGBoost)
    - P4: SMOTE→FS (XGBoost+Wrapper+K30)
    - P5: FS→SMOTE (XGBoost+Wrapper+K30)
    """
    print_section_header(
        "UNSW-NB15核心实验",
        "UNSW-NB15 Core Experiments"
    )
    
    unsw_dir = create_output_subdir(output_dir, '12_unsw_nb15')
    results_list = []
    
    configs = [
        {'pipeline': 'P1', 'model': 'xgboost', 'name': 'P1-基线'},
        {'pipeline': 'P2', 'model': 'xgboost', 'name': 'P2-仅SMOTE'},
        {'pipeline': 'P4', 'model': 'xgboost', 'fs_method': 'wrapper', 'k': 30, 'name': 'P4-SMOTE→FS'},
        {'pipeline': 'P5', 'model': 'xgboost', 'fs_method': 'wrapper', 'k': 30, 'name': 'P5-FS→SMOTE'},
    ]
    
    for config_idx, exp_config in enumerate(configs, 1):
        pipeline = exp_config['pipeline']
        model_name = exp_config['model']
        fs_method = exp_config.get('fs_method')
        k = exp_config.get('k')
        name = exp_config['name']
        
        print(f"\n{'='*60}")
        print(f"[{config_idx}/{len(configs)}] {name}")
        print(f"{'='*60}")
        
        try:
            if pipeline == 'P1':
                # P1: 基线
                model = get_model(model_name)
                
                if model_name == 'xgboost':
                    y_train_model, label_mapping, inverse_mapping = encode_labels_for_xgboost(y_train)
                    model.fit(X_train, y_train_model)
                    y_pred_encoded = model.predict(X_test)
                    y_pred = np.array([inverse_mapping[p] for p in y_pred_encoded])
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                metrics = calculate_metrics(y_test, y_pred, average=None)
                
            elif pipeline == 'P2':
                # P2: 仅SMOTE
                X_resampled, y_resampled, smote_time = apply_smote(X_train, y_train)
                
                model = get_model(model_name)
                
                if model_name == 'xgboost':
                    y_resampled_model, label_mapping, inverse_mapping = encode_labels_for_xgboost(y_resampled)
                    model.fit(X_resampled, y_resampled_model)
                    y_pred_encoded = model.predict(X_test)
                    y_pred = np.array([inverse_mapping[p] for p in y_pred_encoded])
                else:
                    model.fit(X_resampled, y_resampled)
                    y_pred = model.predict(X_test)
                
                metrics = calculate_metrics(y_test, y_pred, average=None)
                metrics['smote_time'] = smote_time
                
            elif pipeline == 'P4':
                # P4: SMOTE→FS
                X_resampled, y_resampled, smote_time = apply_smote(X_train, y_train)
                
                if fs_method == 'filter':
                    X_train_sel, X_test_sel, _, _, fs_time = filter_feature_selection(
                        X_resampled, y_resampled, X_test, k)
                else:
                    X_train_sel, X_test_sel, _, _, fs_time = wrapper_feature_selection(
                        X_resampled, y_resampled, X_test, k)
                
                model = get_model(model_name)
                
                if model_name == 'xgboost':
                    y_resampled_model, label_mapping, inverse_mapping = encode_labels_for_xgboost(y_resampled)
                    model.fit(X_train_sel, y_resampled_model)
                    y_pred_encoded = model.predict(X_test_sel)
                    y_pred = np.array([inverse_mapping[p] for p in y_pred_encoded])
                else:
                    model.fit(X_train_sel, y_resampled)
                    y_pred = model.predict(X_test_sel)
                
                metrics = calculate_metrics(y_test, y_pred, average=None)
                metrics['smote_time'] = smote_time
                metrics['fs_time'] = fs_time
                
            elif pipeline == 'P5':
                # P5: FS→SMOTE
                if fs_method == 'filter':
                    X_train_sel, X_test_sel, _, _, fs_time = filter_feature_selection(
                        X_train, y_train, X_test, k)
                else:
                    X_train_sel, X_test_sel, _, _, fs_time = wrapper_feature_selection(
                        X_train, y_train, X_test, k)
                
                # SMOTE在精简特征空间
                min_class_count = min(Counter(y_train).values())
                k_neighbors = min(config.SMOTE_CONFIG['k_neighbors'], min_class_count - 1)
                k_neighbors = max(k_neighbors, 1)
                
                smote = SMOTE(
                    k_neighbors=k_neighbors,
                    sampling_strategy=config.SMOTE_CONFIG['sampling_strategy'],
                    random_state=config.RANDOM_STATE
                )
                X_resampled, y_resampled = smote.fit_resample(X_train_sel, y_train)
                
                model = get_model(model_name)
                
                if model_name == 'xgboost':
                    y_resampled_model, label_mapping, inverse_mapping = encode_labels_for_xgboost(y_resampled)
                    model.fit(X_resampled, y_resampled_model)
                    y_pred_encoded = model.predict(X_test_sel)
                    y_pred = np.array([inverse_mapping[p] for p in y_pred_encoded])
                else:
                    model.fit(X_resampled, y_resampled)
                    y_pred = model.predict(X_test_sel)
                
                metrics = calculate_metrics(y_test, y_pred, average=None)
                metrics['fs_time'] = fs_time
            
            # 记录结果
            result = {
                'dataset': 'UNSW-NB15',
                'pipeline': pipeline,
                'model': model_name,
                'fs_method': fs_method if fs_method else 'none',
                'fs_k': k if k else 'none',
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
            }
            
            # 添加每类召回率
            for cls in ['normal', 'dos', 'probe', 'r2l', 'u2r']:
                col_name = f'recall_{cls}'
                if col_name in metrics:
                    result[col_name] = metrics[col_name]
            
            results_list.append(result)
            
            print(f"完成: F1-macro={metrics['f1_macro']:.4f}, "
                  f"Accuracy={metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"错误: {name} 实验失败: {e}")
            results_list.append({
                'dataset': 'UNSW-NB15',
                'pipeline': pipeline,
                'model': model_name,
                'error': str(e)
            })
    
    # 保存结果
    results_df = pd.DataFrame(results_list)
    results_file = save_results(results_list, unsw_dir, 'unsw_results.csv')
    print(f"\n结果已保存: {results_file}")
    
    # 绘制对比图（双语）
    plot_unsw_comparison(results_df, unsw_dir, language='ch')
    plot_unsw_comparison(results_df, unsw_dir, language='en')
    
    return results_list


def plot_unsw_comparison(results_df, output_dir, language='ch'):
    """
    绘制UNSW-NB15结果对比图
    """
    if language == 'ch':
        labels = {
            'title': 'UNSW-NB15数据集性能对比',
            'xlabel': '实验配置',
            'ylabel': 'F1-macro',
        }
        filename = 'unsw_comparison_ch.png'
    else:
        labels = {
            'title': 'UNSW-NB15 Dataset Performance Comparison',
            'xlabel': 'Experiment Configuration',
            'ylabel': 'F1-macro',
        }
        filename = 'unsw_comparison_en.png'
    
    if 'error' in results_df.columns:
        valid_df = results_df[results_df['error'].isna()]
    else:
        valid_df = results_df
    
    if len(valid_df) == 0:
        print("警告: 没有有效的实验结果用于绘图")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pipelines = valid_df['pipeline'].values
    f1_scores = valid_df['f1_macro'].values
    
    colors = ['steelblue', 'coral', 'green', 'orange']
    bars = ax.bar(pipelines, f1_scores, color=colors[:len(pipelines)], alpha=0.8)
    
    ax.set_xlabel(labels['xlabel'], fontsize=12)
    ax.set_ylabel(labels['ylabel'], fontsize=12)
    ax.set_title(labels['title'], fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"UNSW对比图已保存: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    print_section_header(
        "阶段12：UNSW-NB15数据集实验",
        "Stage 12: UNSW-NB15 Dataset Experiment"
    )
    
    # 创建输出目录
    output_dir = config.create_output_dir()
    print(f"输出目录: {output_dir}")
    
    # 加载UNSW数据
    train_df, test_df = load_unsw_data()
    
    # 调试模式采样
    if config.DEBUG_MODE:
        print(f"[调试模式] 采样训练集")
        train_df = train_df.sample(n=min(config.DEBUG_TRAIN_SIZE, len(train_df)),
                                   random_state=config.RANDOM_STATE)
        test_df = test_df.sample(n=min(config.DEBUG_TEST_SIZE, len(test_df)),
                                random_state=config.RANDOM_STATE)
    
    # 预处理
    X_train, y_train, X_test, y_test = preprocess_unsw_data(train_df, test_df)
    
    # 运行实验
    results = run_unsw_experiment(X_train, y_train, X_test, y_test, output_dir)
    
    print_section_header(
        "UNSW-NB15实验完成",
        "UNSW-NB15 Experiment Completed"
    )
    
    return results


if __name__ == "__main__":
    main()
