"""
09_超参数调优实验 (Hyperparameter Tuning)

功能:
1. SMOTE k值敏感性分析
2. XGBoost超参数网格搜索

输出:
- 不同k值下的性能曲线
- 最优超参数组合及性能提升
"""

import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
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
from smote_only import apply_smote
from baseline import get_model


def smote_k_sensitivity_analysis(X_train, y_train, X_test, y_test, output_dir, model_name='xgboost'):
    """
    SMOTE k值敏感性分析
    
    参数:
        k_values: k值列表 [3, 5, 7, 10]
        model_name: 测试的模型名称
    """
    print_section_header(
        "SMOTE k值敏感性分析",
        "SMOTE k-value Sensitivity Analysis"
    )
    
    k_dir = create_output_subdir(output_dir, '09_smote_k_sensitivity')
    results_list = []
    
    k_values = [3, 5, 7, 10]
    
    print(f"\n测试k值范围: {k_values}")
    print(f"测试模型: {model_name}")
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"测试 k={k}")
        print(f"{'='*60}")
        
        # 临时修改SMOTE配置
        original_k = config.SMOTE_CONFIG['k_neighbors']
        config.SMOTE_CONFIG['k_neighbors'] = k
        
        try:
            # 应用SMOTE
            X_resampled, y_resampled, smote_time = apply_smote(X_train, y_train)
            
            # 训练模型
            model = get_model(model_name)
            
            start_time = time.time()
            if model_name == 'xgboost':
                y_resampled_model, label_mapping, inverse_mapping = encode_labels_for_xgboost(y_resampled)
                model.fit(X_resampled, y_resampled_model)
                y_pred_encoded = model.predict(X_test)
                y_pred = np.array([inverse_mapping[p] for p in y_pred_encoded])
            else:
                model.fit(X_resampled, y_resampled)
                y_pred = model.predict(X_test)
            
            train_time = time.time() - start_time
            
            # 计算指标
            metrics = calculate_metrics(y_test, y_pred, average=None)
            
            # 记录结果
            result = {
                'k_value': k,
                'model': model_name,
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'recall_r2l': metrics.get('recall_r2l', 0),
                'recall_u2r': metrics.get('recall_u2r', 0),
                'train_time': train_time,
                'smote_time': smote_time,
            }
            results_list.append(result)
            
            print(f"k={k}: F1-macro={metrics['f1_macro']:.4f}, "
                  f"R2L-Recall={metrics.get('recall_r2l', 0):.4f}, "
                  f"U2R-Recall={metrics.get('recall_u2r', 0):.4f}")
            
        finally:
            # 恢复原始配置
            config.SMOTE_CONFIG['k_neighbors'] = original_k
    
    # 保存结果
    results_df = pd.DataFrame(results_list)
    results_file = save_results(results_list, k_dir, 'k_sensitivity_results.csv')
    print(f"\n结果已保存: {results_file}")
    
    # 绘制k值敏感性曲线（双语版本）
    plot_k_sensitivity(results_df, k_dir, language='ch')
    plot_k_sensitivity(results_df, k_dir, language='en')
    
    return results_list


def plot_k_sensitivity(results_df, output_dir, language='ch'):
    """
    绘制SMOTE k值敏感性曲线
    
    参数:
        language: 'ch' 或 'en'
    """
    if language == 'ch':
        labels = {
            'title': 'SMOTE k值敏感性分析',
            'xlabel': 'k值（近邻数量）',
            'ylabel': '性能指标',
            'f1_macro': 'F1-macro',
            'r2l_recall': 'R2L召回率',
            'uur_recall': 'U2R召回率',
        }
        filename = 'k_sensitivity_ch.png'
    else:
        labels = {
            'title': 'SMOTE k-value Sensitivity Analysis',
            'xlabel': 'k Value (Number of Neighbors)',
            'ylabel': 'Performance Metric',
            'f1_macro': 'F1-macro',
            'r2l_recall': 'R2L Recall',
            'uur_recall': 'U2R Recall',
        }
        filename = 'k_sensitivity_en.png'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = results_df['k_value'].values
    
    # 绘制三条曲线
    ax.plot(k_values, results_df['f1_macro'].values, 
            marker='o', linewidth=2, label=labels['f1_macro'], color='steelblue')
    ax.plot(k_values, results_df['recall_r2l'].values, 
            marker='s', linewidth=2, label=labels['r2l_recall'], color='coral')
    ax.plot(k_values, results_df['recall_u2r'].values, 
            marker='^', linewidth=2, label=labels['uur_recall'], color='green')
    
    ax.set_xlabel(labels['xlabel'], fontsize=12)
    ax.set_ylabel(labels['ylabel'], fontsize=12)
    ax.set_title(labels['title'], fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"k值敏感性曲线已保存: {output_path}")
    
    plt.close()


def xgboost_hyperparameter_search(X_train, y_train, X_test, y_test, output_dir, 
                                   pipeline_type='P2', fs_method=None, k=None):
    """
    XGBoost超参数网格搜索
    
    参数:
        pipeline_type: 'P2'(仅SMOTE) 或 'P4'(SMOTE→FS)
        fs_method: 特征选择方法（P4时使用）
        k: 特征数量（P4时使用）
    """
    print_section_header(
        "XGBoost超参数网格搜索",
        "XGBoost Hyperparameter Grid Search"
    )
    
    grid_dir = create_output_subdir(output_dir, '09_xgboost_grid_search')
    
    # 准备数据
    if pipeline_type == 'P2':
        # P2: 仅SMOTE
        X_resampled, y_resampled, _ = apply_smote(X_train, y_train)
        X_train_grid = X_resampled
        y_train_grid = y_resampled
        print("使用P2流水线数据 (SMOTE后)")
    elif pipeline_type == 'P4' and fs_method and k:
        # P4: SMOTE→FS
        from joint_pipeline_smote_fs import run_joint_smote_fs_experiment
        # 这里简化处理，直接使用SMOTE+FS后的数据
        X_resampled, y_resampled, _ = apply_smote(X_train, y_train)
        from feature_selection import filter_feature_selection, wrapper_feature_selection
        if fs_method == 'filter':
            X_train_grid, X_test_grid, _, _, _ = filter_feature_selection(
                X_resampled, y_resampled, X_test, k)
        else:
            X_train_grid, X_test_grid, _, _, _ = wrapper_feature_selection(
                X_resampled, y_resampled, X_test, k)
        y_train_grid = y_resampled
        print(f"使用P4流水线数据 (SMOTE→{fs_method.upper()}, Top-{k})")
    else:
        raise ValueError("参数错误: pipeline_type必须是'P2'或'P4'")
    
    # 为XGBoost编码标签
    y_train_grid_encoded, label_mapping, inverse_mapping = encode_labels_for_xgboost(y_train_grid)
    
    # 定义参数网格（第一阶段：粗网格）
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.1, 0.3],
        'n_estimators': [100, 300]
    }
    
    print(f"\n参数搜索空间:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # 使用F1-macro作为评估指标
    f1_scorer = make_scorer(f1_score, average='macro')
    
    # 创建XGBoost分类器
    xgb_model = xgb.XGBClassifier(
        random_state=config.RANDOM_STATE,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    
    # 执行网格搜索
    print(f"\n开始网格搜索...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train_grid, y_train_grid_encoded)
    
    grid_time = time.time() - start_time
    print(f"\n网格搜索完成！耗时: {grid_time:.2f}秒")
    
    # 保存结果
    results = pd.DataFrame(grid_search.cv_results_)
    results_file = grid_dir / 'grid_search_results.csv'
    results.to_csv(results_file, index=False)
    print(f"网格搜索结果已保存: {results_file}")
    
    # 最优参数
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n最优参数组合:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"最优CV F1-macro: {best_score:.4f}")
    
    # 使用最优参数在测试集上评估
    best_model = grid_search.best_estimator_
    # P4流水线需要使用特征选择后的测试集
    X_test_eval = X_test if pipeline_type == 'P2' else X_test_grid
    y_pred_encoded = best_model.predict(X_test_eval)
    y_pred = np.array([inverse_mapping[p] for p in y_pred_encoded])
    
    test_metrics = calculate_metrics(y_test, y_pred, average=None)
    
    print(f"\n测试集性能:")
    print(f"  F1-macro: {test_metrics['f1_macro']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    # 保存最优参数和性能
    summary = {
        'pipeline': pipeline_type,
        'fs_method': fs_method if fs_method else 'none',
        'fs_k': k if k else 'none',
        'best_params': best_params,
        'best_cv_score': best_score,
        'test_f1_macro': test_metrics['f1_macro'],
        'test_accuracy': test_metrics['accuracy'],
        'grid_search_time': grid_time,
    }
    
    summary_file = grid_dir / 'best_params.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 绘制热力图（如果只有2个参数变化）
    if len(param_grid) == 2:
        plot_grid_heatmap(results, param_grid, grid_dir)
    
    return summary


def plot_grid_heatmap(results_df, param_grid, output_dir):
    """
    绘制网格搜索热力图
    """
    # 获取变化的两个参数
    params = list(param_grid.keys())
    if len(params) != 2:
        return
    
    param1, param2 = params
    
    # 创建透视表
    pivot = results_df.pivot_table(
        values='mean_test_score',
        index=f'param_{param1}',
        columns=f'param_{param2}',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel(param2, fontsize=12)
    ax.set_ylabel(param1, fontsize=12)
    ax.set_title('网格搜索F1-macro热力图', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, f'{pivot.values[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label='F1-macro')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'grid_search_heatmap.png'
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"网格搜索热力图已保存: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    print_section_header(
        "阶段9：超参数调优实验",
        "Stage 9: Hyperparameter Tuning"
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
    
    # 1. SMOTE k值敏感性分析
    print("\n" + "="*60)
    print("1. SMOTE k值敏感性分析")
    print("="*60)
    k_results = smote_k_sensitivity_analysis(X_train, y_train, X_test, y_test, output_dir)
    
    # 2. XGBoost网格搜索（P2）
    print("\n" + "="*60)
    print("2. XGBoost超参数搜索 (P2-SMOTE)")
    print("="*60)
    p2_grid = xgboost_hyperparameter_search(X_train, y_train, X_test, y_test, 
                                             output_dir, pipeline_type='P2')
    
    # 3. XGBoost网格搜索（P4）
    print("\n" + "="*60)
    print("3. XGBoost超参数搜索 (P4-SMOTE→FS)")
    print("="*60)
    p4_grid = xgboost_hyperparameter_search(X_train, y_train, X_test, y_test,
                                             output_dir, pipeline_type='P4',
                                             fs_method='wrapper', k=30)
    
    print_section_header(
        "超参数调优实验完成",
        "Hyperparameter Tuning Completed"
    )
    
    return k_results, p2_grid, p4_grid


if __name__ == "__main__":
    main()
