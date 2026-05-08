"""
11_统计显著性检验 (Statistical Significance Testing)

功能:
对关键实验配置进行5次重复实验，并进行统计检验

统计检验:
1. 配对t检验 (Paired t-test)
2. Wilcoxon符号秩检验 (非参数备用)
3. 效应量计算 (Cohen's d)

输出:
- 重复实验结果（均值±标准差）
- 统计检验结果表
- 带误差棒的性能对比图（中英双语）
"""

import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

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


def run_single_experiment(X_train, y_train, X_test, y_test, 
                          pipeline, model_name, fs_method=None, k=None, 
                          random_state=42):
    """
    运行单次实验
    
    参数:
        pipeline: 流水线类型 ('P1', 'P2', 'P3', 'P4', 'P5')
        model_name: 模型名称
        fs_method: 特征选择方法 ('filter', 'wrapper')
        k: 特征数量
        random_state: 随机种子
    
    返回:
        metrics: 评估指标字典
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 复制配置并修改随机种子
    original_random_state = config.RANDOM_STATE
    config.RANDOM_STATE = random_state
    
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
            metrics['pipeline'] = 'P1'
            
        elif pipeline == 'P2':
            # P2: 仅SMOTE
            X_resampled, y_resampled, _ = apply_smote(X_train, y_train)
            
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
            metrics['pipeline'] = 'P2'
            
        elif pipeline == 'P3':
            # P3: 仅特征选择
            if fs_method == 'filter':
                X_train_sel, X_test_sel, _, _, _ = filter_feature_selection(
                    X_train, y_train, X_test, k)
            else:
                X_train_sel, X_test_sel, _, _, _ = wrapper_feature_selection(
                    X_train, y_train, X_test, k)
            
            model = get_model(model_name)
            
            if model_name == 'xgboost':
                y_train_model, label_mapping, inverse_mapping = encode_labels_for_xgboost(y_train)
                model.fit(X_train_sel, y_train_model)
                y_pred_encoded = model.predict(X_test_sel)
                y_pred = np.array([inverse_mapping[p] for p in y_pred_encoded])
            else:
                model.fit(X_train_sel, y_train)
                y_pred = model.predict(X_test_sel)
            
            metrics = calculate_metrics(y_test, y_pred, average=None)
            metrics['pipeline'] = f'P3_{fs_method}_top{k}'
            
        elif pipeline == 'P4':
            # P4: SMOTE→FS
            X_resampled, y_resampled, _ = apply_smote(X_train, y_train)
            
            if fs_method == 'filter':
                X_train_sel, X_test_sel, _, _, _ = filter_feature_selection(
                    X_resampled, y_resampled, X_test, k)
            else:
                X_train_sel, X_test_sel, _, _, _ = wrapper_feature_selection(
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
            metrics['pipeline'] = f'P4_{fs_method}_top{k}'
            
        elif pipeline == 'P5':
            # P5: FS→SMOTE
            if fs_method == 'filter':
                X_train_sel, X_test_sel, _, _, _ = filter_feature_selection(
                    X_train, y_train, X_test, k)
            else:
                X_train_sel, X_test_sel, _, _, _ = wrapper_feature_selection(
                    X_train, y_train, X_test, k)
            
            # 在精简特征空间上应用SMOTE
            min_class_count = min(pd.Series(y_train).value_counts())
            k_neighbors = min(config.SMOTE_CONFIG['k_neighbors'], min_class_count - 1)
            k_neighbors = max(k_neighbors, 1)
            
            smote = SMOTE(
                k_neighbors=k_neighbors,
                sampling_strategy=config.SMOTE_CONFIG['sampling_strategy'],
                random_state=random_state
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
            metrics['pipeline'] = f'P5_{fs_method}_top{k}'
        
        metrics['model'] = model_name
        metrics['random_state'] = random_state
        
        return metrics
        
    finally:
        # 恢复原始随机种子
        config.RANDOM_STATE = original_random_state


def run_repeated_experiments(X_train, y_train, X_test, y_test, 
                              experiment_configs, n_repeats=5, 
                              random_states=None, output_dir=None):
    """
    运行重复实验
    
    参数:
        experiment_configs: 实验配置列表 [{'pipeline': 'P1', 'model': 'xgboost', ...}]
        n_repeats: 重复次数
        random_states: 随机种子列表
        output_dir: 输出目录
    
    返回:
        results_df: 所有重复实验结果
    """
    print_section_header(
        f"统计显著性检验 ({n_repeats}次重复实验)",
        f"Statistical Significance Testing ({n_repeats} Repeats)"
    )
    
    if random_states is None:
        random_states = config.REPEAT_EXPERIMENTS['random_states']
    
    stat_dir = create_output_subdir(output_dir, '11_statistical_test')
    
    all_results = []
    total_experiments = len(experiment_configs) * n_repeats
    exp_counter = 0
    
    print(f"\n总实验数: {total_experiments}")
    print(f"实验配置数: {len(experiment_configs)}")
    print(f"重复次数: {n_repeats}")
    print(f"随机种子: {random_states}")
    
    for config_idx, exp_config in enumerate(experiment_configs, 1):
        pipeline = exp_config['pipeline']
        model_name = exp_config['model']
        fs_method = exp_config.get('fs_method')
        k = exp_config.get('k')
        
        print(f"\n{'='*60}")
        print(f"配置 {config_idx}/{len(experiment_configs)}: {pipeline}-{model_name}")
        if fs_method:
            print(f"  FS方法: {fs_method}, K={k}")
        print(f"{'='*60}")
        
        for repeat_idx, seed in enumerate(random_states, 1):
            exp_counter += 1
            print(f"\n[{exp_counter}/{total_experiments}] 重复 {repeat_idx}/{n_repeats} (seed={seed})")
            
            try:
                metrics = run_single_experiment(
                    X_train, y_train, X_test, y_test,
                    pipeline, model_name, fs_method, k, seed
                )
                
                metrics['repeat_id'] = repeat_idx
                metrics['config_id'] = config_idx
                all_results.append(metrics)
                
                print(f"  F1-macro: {metrics['f1_macro']:.4f}, "
                      f"Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                print(f"  错误: {e}")
                all_results.append({
                    'config_id': config_idx,
                    'repeat_id': repeat_idx,
                    'pipeline': pipeline,
                    'model': model_name,
                    'error': str(e)
                })
    
    # 保存原始结果
    results_df = pd.DataFrame(all_results)
    results_file = save_results(all_results, stat_dir, 'repeated_experiment_results.csv')
    print(f"\n原始结果已保存: {results_file}")
    
    return results_df, stat_dir


def calculate_statistics(results_df, output_dir):
    """
    计算描述性统计和统计检验
    
    参数:
        results_df: 重复实验结果DataFrame
        output_dir: 输出目录
    
    返回:
        stats_summary: 统计摘要
    """
    print_section_header(
        "统计分析与假设检验",
        "Statistical Analysis and Hypothesis Testing"
    )
    
    # 过滤掉错误结果
    if 'error' in results_df.columns:
        valid_df = results_df[results_df['error'].isna()]
    else:
        valid_df = results_df
    
    if len(valid_df) == 0:
        print("错误: 没有有效的实验结果")
        return None
    
    # 1. 描述性统计
    print("\n1. 描述性统计 (均值±标准差)")
    print("="*60)
    
    # 按配置分组计算统计量
    group_cols = ['pipeline', 'model']
    if 'fs_method' in valid_df.columns:
        group_cols.append('fs_method')
    if 'k' in valid_df.columns:
        group_cols.append('k')
    
    desc_stats = valid_df.groupby(group_cols).agg({
        'f1_macro': ['mean', 'std', 'min', 'max'],
        'accuracy': ['mean', 'std'],
        'recall_r2l': ['mean', 'std'],
        'recall_u2r': ['mean', 'std'],
    }).round(4)
    
    print(desc_stats)
    
    # 保存描述性统计
    desc_file = Path(output_dir) / 'descriptive_statistics.csv'
    desc_stats.to_csv(desc_file)
    print(f"\n描述性统计已保存: {desc_file}")
    
    # 2. 配对t检验
    print("\n2. 配对t检验")
    print("="*60)
    
    # 定义对比组
    comparisons = [
        ('P2_xgboost', 'P1_xgboost', 'P2 vs P1 (SMOTE vs Baseline)'),
        ('P4_wrapper_top30_xgboost', 'P2_xgboost', 'P4 vs P2 (Joint vs SMOTE)'),
        ('P5_wrapper_top30_xgboost', 'P4_wrapper_top30_xgboost', 'P5 vs P4 (Order Effect)'),
    ]
    
    ttest_results = []
    
    for group1_key, group2_key, description in comparisons:
        # 提取两组数据
        group1 = valid_df[valid_df['pipeline'] == group1_key.split('_')[0]]
        group2 = valid_df[valid_df['pipeline'] == group2_key.split('_')[0]]
        
        if len(group1) == 0 or len(group2) == 0:
            print(f"  跳过 {description}: 数据不足")
            continue
        
        # 按repeat_id配对
        paired_data = []
        for repeat_id in valid_df['repeat_id'].unique():
            g1 = group1[group1['repeat_id'] == repeat_id]['f1_macro'].values
            g2 = group2[group2['repeat_id'] == repeat_id]['f1_macro'].values
            if len(g1) > 0 and len(g2) > 0:
                paired_data.append((g1[0], g2[0]))
        
        if len(paired_data) < 3:
            print(f"  跳过 {description}: 配对样本不足 ({len(paired_data)}对)")
            continue
        
        g1_values = [x[0] for x in paired_data]
        g2_values = [x[1] for x in paired_data]
        
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(g1_values, g2_values)
        
        # 效应量 (Cohen's d)
        diff = np.array(g1_values) - np.array(g2_values)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        
        # Wilcoxon检验 (非参数备用)
        try:
            w_stat, w_pvalue = stats.wilcoxon(g1_values, g2_values)
        except:
            w_stat, w_pvalue = None, None
        
        result = {
            'comparison': description,
            'n_pairs': len(paired_data),
            'group1_mean': np.mean(g1_values),
            'group2_mean': np.mean(g2_values),
            'difference': np.mean(g1_values) - np.mean(g2_values),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'wilcoxon_stat': w_stat,
            'wilcoxon_p': w_pvalue,
            'significant_0.05': p_value < 0.05,
            'significant_0.01': p_value < 0.01,
        }
        ttest_results.append(result)
        
        print(f"\n  {description}:")
        print(f"    配对样本数: {len(paired_data)}")
        print(f"    Group1均值: {np.mean(g1_values):.4f} ± {np.std(g1_values):.4f}")
        print(f"    Group2均值: {np.mean(g2_values):.4f} ± {np.std(g2_values):.4f}")
        print(f"    差值: {np.mean(g1_values) - np.mean(g2_values):.4f}")
        print(f"    t统计量: {t_stat:.4f}")
        print(f"    p值: {p_value:.4f}")
        print(f"    Cohen's d: {cohens_d:.4f}")
        if w_pvalue is not None:
            print(f"    Wilcoxon p: {w_pvalue:.4f}")
        print(f"    显著性(α=0.05): {'是' if p_value < 0.05 else '否'}")
    
    # 保存统计检验结果
    ttest_df = pd.DataFrame(ttest_results)
    ttest_file = Path(output_dir) / 'statistical_tests.csv'
    ttest_df.to_csv(ttest_file, index=False)
    print(f"\n统计检验结果已保存: {ttest_file}")
    
    # 3. 生成统计报告
    report = generate_statistical_report(ttest_results, desc_stats, output_dir)
    
    return desc_stats, ttest_df


def generate_statistical_report(ttest_results, desc_stats, output_dir):
    """
    生成统计报告文本
    """
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("统计显著性检验报告")
    report_lines.append("Statistical Significance Testing Report")
    report_lines.append("="*70)
    report_lines.append("")
    
    report_lines.append("1. 实验设计")
    report_lines.append("-"*70)
    report_lines.append(f"重复次数: 5次")
    report_lines.append(f"随机种子: {config.REPEAT_EXPERIMENTS['random_states']}")
    report_lines.append("")
    
    report_lines.append("2. 描述性统计")
    report_lines.append("-"*70)
    report_lines.append(str(desc_stats))
    report_lines.append("")
    
    report_lines.append("3. 假设检验结果")
    report_lines.append("-"*70)
    
    for result in ttest_results:
        report_lines.append(f"\n{result['comparison']}:")
        report_lines.append(f"  配对样本数: {result['n_pairs']}")
        report_lines.append(f"  均值差: {result['difference']:.4f}")
        report_lines.append(f"  t = {result['t_statistic']:.4f}, p = {result['p_value']:.4f}")
        report_lines.append(f"  Cohen's d = {result['cohens_d']:.4f}")
        report_lines.append(f"  显著性(α=0.05): {'***' if result['significant_0.01'] else ('**' if result['significant_0.05'] else 'ns')}")
    
    report_lines.append("")
    report_lines.append("4. 结论")
    report_lines.append("-"*70)
    
    for result in ttest_results:
        if result['significant_0.01']:
            sig_level = "高度显著 (p<0.01)"
        elif result['significant_0.05']:
            sig_level = "显著 (p<0.05)"
        else:
            sig_level = "不显著 (p≥0.05)"
        
        report_lines.append(f"{result['comparison']}: {sig_level}")
    
    report_lines.append("")
    report_lines.append("="*70)
    
    report_text = "\n".join(report_lines)
    
    # 保存报告
    report_file = Path(output_dir) / 'statistical_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n统计报告已保存: {report_file}")
    
    return report_text


def plot_repeated_experiments(results_df, output_dir, language='ch'):
    """
    绘制带误差棒的性能对比图
    """
    if language == 'ch':
        labels = {
            'title': '重复实验性能对比 (均值±标准差)',
            'xlabel': '实验配置',
            'ylabel': 'F1-macro',
        }
        filename = 'repeated_experiments_errorbar_ch.png'
    else:
        labels = {
            'title': 'Repeated Experiments Performance (Mean±Std)',
            'xlabel': 'Experiment Configuration',
            'ylabel': 'F1-macro',
        }
        filename = 'repeated_experiments_errorbar_en.png'
    
    if 'error' in results_df.columns:
        valid_df = results_df[results_df['error'].isna()]
    else:
        valid_df = results_df
    
    if len(valid_df) == 0:
        print("警告: 没有有效的实验结果用于绘图")
        return
    
    # 计算均值和标准差
    group_cols = ['pipeline', 'model']
    if 'fs_method' in valid_df.columns:
        group_cols.append('fs_method')
    if 'k' in valid_df.columns:
        group_cols.append('k')
    
    summary = valid_df.groupby(group_cols)['f1_macro'].agg(['mean', 'std']).reset_index()
    summary['label'] = summary.apply(
        lambda x: f"{x['pipeline']}-{x['model']}", axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(summary))
    bars = ax.bar(x, summary['mean'].values, 
                   yerr=summary['std'].values,
                   capsize=5, color='steelblue', alpha=0.8,
                   error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    ax.set_xlabel(labels['xlabel'], fontsize=12)
    ax.set_ylabel(labels['ylabel'], fontsize=12)
    ax.set_title(labels['title'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['label'].values, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (mean, std) in enumerate(zip(summary['mean'].values, summary['std'].values)):
        ax.text(i, mean + std + 0.005, f'{mean:.3f}±{std:.3f}',
               ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"带误差棒的对比图已保存: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    print_section_header(
        "阶段11：统计显著性检验",
        "Stage 11: Statistical Significance Testing"
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
    
    # 定义关键实验配置
    experiment_configs = [
        {'pipeline': 'P1', 'model': 'xgboost'},
        {'pipeline': 'P2', 'model': 'xgboost'},
        {'pipeline': 'P3', 'model': 'xgboost', 'fs_method': 'wrapper', 'k': 30},
        {'pipeline': 'P4', 'model': 'xgboost', 'fs_method': 'wrapper', 'k': 30},
        {'pipeline': 'P5', 'model': 'xgboost', 'fs_method': 'wrapper', 'k': 30},
        {'pipeline': 'P2', 'model': 'random_forest'},
        {'pipeline': 'P4', 'model': 'random_forest', 'fs_method': 'wrapper', 'k': 30},
    ]
    
    # 运行重复实验
    results_df, stat_dir = run_repeated_experiments(
        X_train, y_train, X_test, y_test,
        experiment_configs, 
        n_repeats=config.REPEAT_EXPERIMENTS['n_repeats'],
        random_states=config.REPEAT_EXPERIMENTS['random_states'],
        output_dir=output_dir
    )
    
    # 统计分析
    desc_stats, ttest_df = calculate_statistics(results_df, stat_dir)
    
    # 绘制图表（双语）
    plot_repeated_experiments(results_df, stat_dir, language='ch')
    plot_repeated_experiments(results_df, stat_dir, language='en')
    
    print_section_header(
        "统计显著性检验完成",
        "Statistical Significance Testing Completed"
    )
    
    return results_df, ttest_df


if __name__ == "__main__":
    main()
