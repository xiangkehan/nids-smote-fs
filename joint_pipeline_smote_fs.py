"""
05_联合流水线: SMOTE → 特征选择 (P4)

功能:
先对数据进行SMOTE过采样，然后在过采样后的数据上进行特征选择

流水线:
原始数据 → SMOTE过采样 → 特征选择(Top-K) → 模型训练 → 评估

输出:
- 各模型性能指标
- 与P5(FS→SMOTE)的对比数据
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from collections import Counter

# 导入可视化配置
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


def apply_smote(X_train, y_train):
    """应用SMOTE过采样（自动调整k_neighbors）"""
    print(f"\n步骤1: SMOTE过采样")
    print(f"原始训练集分布:")
    for cls, count in sorted(Counter(y_train).items()):
        print(f"  {cls}: {count}")
    
    # 自动调整k_neighbors：不能超过最小类样本数-1
    min_class_count = min(Counter(y_train).values())
    k_neighbors = min(config.SMOTE_CONFIG['k_neighbors'], min_class_count - 1)
    k_neighbors = max(k_neighbors, 1)  # 至少为1
    
    if k_neighbors < config.SMOTE_CONFIG['k_neighbors']:
        print(f"警告: 最小类只有{min_class_count}个样本，自动调整k_neighbors从{config.SMOTE_CONFIG['k_neighbors']}到{k_neighbors}")
    
    smote = SMOTE(
        k_neighbors=k_neighbors,
        sampling_strategy=config.SMOTE_CONFIG['sampling_strategy'],
        random_state=config.SMOTE_CONFIG['random_state']
    )
    
    start_time = time.time()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    smote_time = time.time() - start_time
    
    print(f"\nSMOTE完成！耗时: {smote_time:.2f}秒")
    print(f"过采样后分布:")
    for cls, count in sorted(Counter(y_resampled).items()):
        print(f"  {cls}: {count}")
    print(f"训练集从 {len(y_train)} 扩展到 {len(y_resampled)} 样本")
    
    return X_resampled, y_resampled, smote_time


def select_features_filter(X, y, k=20):
    """Filter特征选择"""
    print(f"\n步骤2: Filter特征选择 (Top-{k})")
    
    start_time = time.time()
    scores = mutual_info_classif(X, y, random_state=config.RANDOM_STATE)
    fs_time = time.time() - start_time
    
    feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
    
    feature_scores = pd.DataFrame({
        '特征名': feature_names,
        '得分': scores
    })
    feature_scores = feature_scores.sort_values('得分', ascending=False)
    
    selected_features = feature_scores.head(k)['特征名'].tolist()
    
    if hasattr(X, 'columns'):
        X_selected = X[selected_features]
    else:
        top_k_indices = np.argsort(scores)[-k:]
        X_selected = X[:, top_k_indices]
    
    print(f"特征选择完成！耗时: {fs_time:.2f}秒")
    print(f"从 {X.shape[1]} 个特征中选择 {k} 个")
    
    return X_selected, selected_features, feature_scores, fs_time


def select_features_wrapper(X, y, k=20):
    """Wrapper特征选择"""
    print(f"\n步骤2: Wrapper特征选择 (Top-{k})")
    
    estimator = RandomForestClassifier(
        n_estimators=50,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    rfe = RFE(estimator=estimator, n_features_to_select=k, step=0.1)
    
    start_time = time.time()
    rfe.fit(X, y)
    fs_time = time.time() - start_time
    
    feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
    selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
    
    X_selected = rfe.transform(X)
    if hasattr(X, 'columns'):
        X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    print(f"特征选择完成！耗时: {fs_time:.2f}秒")
    print(f"从 {X.shape[1]} 个特征中选择 {k} 个")
    
    return X_selected, selected_features, rfe, fs_time


def run_joint_smote_fs_experiment(X_train, y_train, X_test, y_test, output_dir,
                                  fs_method='filter', k=20):
    """
    运行SMOTE→FS联合流水线实验
    
    流水线: 原始数据 → SMOTE → 特征选择 → 模型训练 → 评估
    """
    print_section_header(
        f"P4-SMOTE→特征选择联合流水线 ({fs_method.upper()}, Top-{k})",
        "先过采样，再特征选择"
    )
    
    pipeline_dir = create_output_subdir(output_dir, f'05_joint_smote_fs_{fs_method}_top{k}')
    results_list = []
    
    # 步骤1: SMOTE过采样
    X_resampled, y_resampled, smote_time = apply_smote(X_train, y_train)
    
    # 步骤2: 特征选择（在过采样后的数据上）
    if fs_method == 'filter':
        X_train_selected, selected_features, feature_scores, fs_time = \
            select_features_filter(X_resampled, y_resampled, k)
        
        # 保存特征得分
        feature_scores.to_csv(pipeline_dir / 'feature_scores.csv', index=False)
        
    elif fs_method == 'wrapper':
        X_train_selected, selected_features, rfe, fs_time = \
            select_features_wrapper(X_resampled, y_resampled, k)
    else:
        raise ValueError(f"未知的特征选择方法: {fs_method}")
    
    # 对测试集应用相同的特征选择
    if hasattr(X_test, 'columns'):
        X_test_selected = X_test[selected_features]
    else:
        # 如果是numpy数组，需要重新计算特征选择
        # 这里简化处理，假设测试集也是DataFrame
        X_test_selected = X_test
    
    # 保存选择的特征
    with open(pipeline_dir / 'selected_features.txt', 'w', encoding='utf-8') as f:
        f.write(f"流水线: P4-SMOTE→FS\n")
        f.write(f"特征选择方法: {fs_method}\n")
        f.write(f"选择特征数: {k}\n")
        f.write(f"原始特征数: {X_train.shape[1]}\n")
        f.write(f"过采样后样本数: {len(y_resampled)}\n")
        f.write("\n选择的特征列表:\n")
        for i, feat in enumerate(selected_features, 1):
            f.write(f"{i}. {feat}\n")
    
    # 步骤3: 训练模型
    active_models = config.get_active_models()
    total_models = len(active_models)
    
    print(f"\n步骤3: 训练 {total_models} 个模型")
    
    for idx, (model_name, model_config) in enumerate(active_models.items(), 1):
        print(f"\n[{idx}/{total_models}] 训练模型: {model_name}")
        
        log_experiment_start(
            pipeline_name=f'P4_smote_fs_{fs_method}_top{k}',
            model_name=model_name,
            train_samples_original=len(y_train),
            train_samples_resampled=len(y_resampled),
            test_samples=len(y_test),
            features_selected=k
        )
        
        # 获取模型
        model = get_model(model_name)
        
        # 如果是XGBoost，需要临时编码标签
        y_resampled_model = y_resampled
        if model_name == 'xgboost':
            print(f"  为XGBoost编码标签...")
            y_resampled_model, label_mapping, inverse_mapping = encode_labels_for_xgboost(y_resampled)
        
        # 训练
        start_time = time.time()
        print(f"  训练中...")
        model.fit(X_train_selected, y_resampled_model)
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
        metrics['pipeline'] = f'P4_smote_fs_{fs_method}_top{k}'
        metrics['model'] = model_name
        metrics['fs_method'] = fs_method
        metrics['fs_k'] = k
        metrics['smote_time'] = smote_time
        metrics['fs_time'] = fs_time
        metrics['train_time'] = train_time
        metrics['total_preprocessing_time'] = smote_time + fs_time
        metrics['train_samples_original'] = len(y_train)
        metrics['train_samples_resampled'] = len(y_resampled)
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
        cm_df.to_csv(pipeline_dir / f'confusion_matrix_{model_name}.csv')
        
        # 保存模型
        if config.OUTPUT['save_models']:
            save_model(model, pipeline_dir, f'model_{model_name}.pkl')
        
        # 记录实验结束
        log_experiment_end(f'P4_smote_fs_{fs_method}_top{k}', model_name, metrics, train_time)
    
    # 保存所有结果
    results_df = pd.DataFrame(results_list)
    results_file = save_results(results_list, pipeline_dir, 'results.csv')
    print(f"\n所有结果已保存: {results_file}")
    
    # 打印汇总表
    print("\n" + "=" * 60)
    print("P4-SMOTE→FS实验结果汇总")
    print("=" * 60)
    summary = results_df[['model', 'accuracy', 'f1_macro', 'f1_weighted', 'train_time']]
    print(summary.to_string(index=False))
    
    return results_list


def main():
    """主函数"""
    print_section_header(
        "阶段5：SMOTE→特征选择联合流水线",
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
    
    # 运行实验
    all_results = []
    fs_methods = config.get_active_fs_methods()
    
    for fs_name, fs_config in fs_methods.items():
        thresholds = fs_config['thresholds']
        
        for k in thresholds:
            print(f"\n{'='*60}")
            print(f"运行 P4-SMOTE→FS ({fs_name.upper()}, Top-{k})")
            print(f"{'='*60}")
            
            results = run_joint_smote_fs_experiment(
                X_train, y_train, X_test, y_test,
                output_dir, fs_method=fs_name, k=k
            )
            all_results.extend(results)
    
    print_section_header(
        "P4-SMOTE→FS实验完成",
        ""
    )
    
    return all_results


if __name__ == "__main__":
    main()
