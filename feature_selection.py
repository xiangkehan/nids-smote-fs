"""
04_特征选择实验 (P3-Feature Selection Only)

功能:
先对数据进行特征选择，然后在精简特征上训练模型

流水线:
原始数据 → 特征选择(Top-K) → 模型训练 → 评估

特征选择方法:
1. Filter方法: 基于互信息(Mutual Information)
2. Wrapper方法: 基于递归特征消除(RFE)

输出:
- 选择的特征列表
- 各模型性能指标
- 特征重要性可视化
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
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.metrics import confusion_matrix
import xgboost as xgb

# 导入可视化配置（设置中文字体）
import plot_config

import config
from utils import (
    calculate_metrics, save_results, save_model,
    log_experiment_start, log_experiment_end,
    create_output_subdir, print_section_header, encode_labels_for_xgboost
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


def filter_feature_selection(X_train, y_train, X_test, k=20):
    """
    Filter特征选择：基于互信息
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        k: 选择的特征数量
    
    返回:
        X_train_selected, X_test_selected, selected_features, scores
    """
    print(f"\nFilter特征选择 (互信息)...")
    print(f"选择Top-{k}个特征")
    
    # 计算互信息
    start_time = time.time()
    scores = mutual_info_classif(X_train, y_train, random_state=config.RANDOM_STATE)
    fs_time = time.time() - start_time
    
    # 获取特征名
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # 创建特征得分表
    feature_scores = pd.DataFrame({
        '特征名': feature_names,
        '得分': scores
    })
    feature_scores = feature_scores.sort_values('得分', ascending=False)
    
    # 选择Top-K特征
    selected_features = feature_scores.head(k)['特征名'].tolist()
    
    # 选择特征
    if hasattr(X_train, 'columns'):
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
    else:
        top_k_indices = np.argsort(scores)[-k:]
        X_train_selected = X_train[:, top_k_indices]
        X_test_selected = X_test[:, top_k_indices]
    
    print(f"特征选择完成！耗时: {fs_time:.2f}秒")
    print(f"从 {X_train.shape[1]} 个特征中选择 {k} 个")
    print(f"前5个重要特征:")
    for i, row in feature_scores.head(5).iterrows():
        print(f"  {row['特征名']}: {row['得分']:.4f}")
    
    return X_train_selected, X_test_selected, selected_features, feature_scores, fs_time


def wrapper_feature_selection(X_train, y_train, X_test, k=20):
    """
    Wrapper特征选择：基于递归特征消除(RFE)
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        k: 选择的特征数量
    
    返回:
        X_train_selected, X_test_selected, selected_features, rfe
    """
    print(f"\nWrapper特征选择 (RFE)...")
    print(f"选择Top-{k}个特征")
    
    # 使用随机森林作为基础估计器
    estimator = RandomForestClassifier(
        n_estimators=50,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    # 创建RFE
    rfe = RFE(estimator=estimator, n_features_to_select=k, step=0.1)
    
    # 拟合RFE
    start_time = time.time()
    rfe.fit(X_train, y_train)
    fs_time = time.time() - start_time
    
    # 获取特征名
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # 获取选择的特征
    selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
    
    # 转换数据
    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)
    
    # 创建DataFrame（保持特征名）
    if hasattr(X_train, 'columns'):
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
    
    # 获取特征排名
    feature_ranks = pd.DataFrame({
        '特征名': feature_names,
        '排名': rfe.ranking_,
        '是否选中': rfe.support_
    })
    feature_ranks = feature_ranks.sort_values('排名')
    
    print(f"特征选择完成！耗时: {fs_time:.2f}秒")
    print(f"从 {X_train.shape[1]} 个特征中选择 {k} 个")
    print(f"前5个重要特征:")
    for i, row in feature_ranks.head(5).iterrows():
        print(f"  {row['特征名']}: 排名={row['排名']}")
    
    return X_train_selected, X_test_selected, selected_features, feature_ranks, fs_time


def plot_feature_importance(feature_scores, output_dir, filename='feature_importance.png', top_n=20):
    """
    绘制特征重要性图（纯中文）
    """
    # 取前top_n个特征
    top_features = feature_scores.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 水平条形图
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['得分'].values, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['特征名'].values, fontsize=10)
    ax.invert_yaxis()  # 最重要的在顶部
    ax.set_xlabel('重要性得分', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    ax.set_title(f'特征重要性排序 (Top-{top_n})', fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['得分'], i, f' {row["得分"]:.4f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"特征重要性图已保存: {output_path}")
    
    plt.close()


def run_feature_selection_experiment(X_train, y_train, X_test, y_test, output_dir, 
                                    fs_method='filter', k=20):
    """
    运行特征选择实验
    
    流水线: 原始数据 → 特征选择(Top-K) → 模型训练 → 评估
    
    参数:
        fs_method: 特征选择方法 ('filter' 或 'wrapper')
        k: 选择的特征数量
    """
    print_section_header(
        f"P3-特征选择实验 ({fs_method.upper()}, Top-{k})",
        f"在精简特征上训练模型"
    )
    
    fs_dir = create_output_subdir(output_dir, f'04_feature_selection_{fs_method}_top{k}')
    results_list = []
    
    # 1. 特征选择
    if fs_method == 'filter':
        X_train_selected, X_test_selected, selected_features, feature_scores, fs_time = \
            filter_feature_selection(X_train, y_train, X_test, k)
        
        # 保存特征得分
        feature_scores.to_csv(fs_dir / 'feature_scores.csv', index=False)
        
        # 可视化特征重要性
        plot_feature_importance(feature_scores, fs_dir, 
                              f'feature_importance_top{k}.png', top_n=k)
        
    elif fs_method == 'wrapper':
        X_train_selected, X_test_selected, selected_features, feature_ranks, fs_time = \
            wrapper_feature_selection(X_train, y_train, X_test, k)
        
        # 保存特征排名
        feature_ranks.to_csv(fs_dir / 'feature_ranks.csv', index=False)
    else:
        raise ValueError(f"未知的特征选择方法: {fs_method}")
    
    # 保存选择的特征列表
    with open(fs_dir / 'selected_features.txt', 'w', encoding='utf-8') as f:
        f.write(f"特征选择方法: {fs_method}\n")
        f.write(f"选择特征数: {k}\n")
        f.write(f"原始特征数: {X_train.shape[1]}\n")
        f.write("\n选择的特征列表:\n")
        for i, feat in enumerate(selected_features, 1):
            f.write(f"{i}. {feat}\n")
    
    print(f"\n选择的特征已保存")
    
    # 2. 在选择的特征上训练模型
    active_models = config.get_active_models()
    total_models = len(active_models)
    
    print(f"\n将训练 {total_models} 个模型")
    
    for idx, (model_name, model_config) in enumerate(active_models.items(), 1):
        print(f"\n[{idx}/{total_models}] 训练模型: {model_name}")
        
        log_experiment_start(
            pipeline_name=f'P3_{fs_method}_top{k}',
            model_name=model_name,
            train_samples=len(y_train),
            test_samples=len(y_test),
            features=k
        )
        
        # 获取模型
        model = get_model(model_name)
        
        # 如果是XGBoost，需要临时编码标签
        y_train_model = y_train
        y_test_eval = y_test
        if model_name == 'xgboost':
            print(f"  为XGBoost编码标签...")
            y_train_model, label_mapping, inverse_mapping = encode_labels_for_xgboost(y_train)
        
        # 训练
        start_time = time.time()
        print(f"  训练中...")
        model.fit(X_train_selected, y_train_model)
        train_time = time.time() - start_time
        
        # 预测
        print(f"  预测中...")
        y_pred = model.predict(X_test_selected)
        
        # 如果是XGBoost，将预测结果转换回原始标签
        if model_name == 'xgboost':
            y_pred = np.array([inverse_mapping[p] for p in y_pred])
        
        # 计算指标
        print(f"  计算指标...")
        metrics = calculate_metrics(y_test, y_pred, average=None)
        
        # 添加实验信息
        metrics['pipeline'] = f'P3_{fs_method}_top{k}'
        metrics['model'] = model_name
        metrics['fs_method'] = fs_method
        metrics['fs_k'] = k
        metrics['fs_time'] = fs_time
        metrics['train_time'] = train_time
        metrics['train_samples'] = len(y_train)
        metrics['test_samples'] = len(y_test)
        metrics['num_features_original'] = X_train.shape[1]
        metrics['num_features_selected'] = k
        
        # 保存结果
        results_list.append(metrics)
        
        # 保存混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm,
                            index=sorted(np.unique(y_test)),
                            columns=sorted(np.unique(y_test)))
        cm_df.to_csv(fs_dir / f'confusion_matrix_{model_name}.csv')
        
        # 保存模型
        if config.OUTPUT['save_models']:
            save_model(model, fs_dir, f'model_{model_name}.pkl')
        
        # 记录实验结束
        log_experiment_end(f'P3_{fs_method}_top{k}', model_name, metrics, train_time)
    
    # 保存所有结果
    results_df = pd.DataFrame(results_list)
    results_file = save_results(results_list, fs_dir, 'results.csv')
    print(f"\n所有结果已保存: {results_file}")
    
    # 打印汇总表
    print("\n" + "=" * 60)
    print("特征选择实验结果汇总")
    print("=" * 60)
    summary = results_df[['model', 'accuracy', 'f1_macro', 'f1_weighted', 'train_time']]
    print(summary.to_string(index=False))
    
    return results_list


def main():
    """主函数"""
    print_section_header(
        "阶段4：特征选择实验",
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
    
    # 运行不同特征选择方法的实验
    all_results = []
    
    # 获取启用的特征选择方法
    fs_methods = config.get_active_fs_methods()
    
    for fs_name, fs_config in fs_methods.items():
        thresholds = fs_config['thresholds']
        
        for k in thresholds:
            print(f"\n{'='*60}")
            print(f"运行 {fs_name.upper()} 特征选择 (Top-{k})")
            print(f"{'='*60}")
            
            results = run_feature_selection_experiment(
                X_train, y_train, X_test, y_test, 
                output_dir, fs_method=fs_name, k=k
            )
            all_results.extend(results)
    
    print_section_header(
        "特征选择实验完成",
        ""
    )
    
    return all_results


if __name__ == "__main__":
    main()
