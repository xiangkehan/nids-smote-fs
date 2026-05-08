"""
02_基线实验 (P1-Baseline)

功能:
在原始数据上直接训练3种模型，作为性能基准

流水线:
原始数据 → 模型训练 → 评估

输出:
- 各模型性能指标
- 混淆矩阵
- 每类详细指标
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import config
from utils import (
    calculate_metrics, save_results, save_model, save_checkpoint,
    load_checkpoint, log_experiment_start, log_experiment_end,
    create_output_subdir, print_section_header, encode_labels_for_xgboost
)
from sklearn.metrics import confusion_matrix


def get_model(model_name):
    """
    根据名称获取模型实例
    
    参数:
        model_name: 模型名称
    
    返回:
        model: 模型实例
    """
    models_config = config.get_active_models()
    
    if model_name not in models_config:
        raise ValueError(f"未知模型 | Unknown model: {model_name}")
    
    model_config = models_config[model_name]
    params = model_config['params']
    
    if model_name == 'decision_tree':
        return DecisionTreeClassifier(**params)
    elif model_name == 'random_forest':
        return RandomForestClassifier(**params)
    elif model_name == 'xgboost':
        return xgb.XGBClassifier(**params)
    else:
        raise ValueError(f"未实现的模型 | Unimplemented model: {model_name}")


def run_baseline_experiment(X_train, y_train, X_test, y_test, output_dir):
    """
    运行基线实验
    
    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        output_dir: 输出目录
    
    返回:
        results_list: 所有实验结果列表
    """
    print_section_header(
        "P1-基线实验 | P1-Baseline Experiment",
        "在原始数据上直接训练模型 | Train models on raw data"
    )
    
    baseline_dir = create_output_subdir(output_dir, '02_baseline')
    results_list = []
    
    # 记录流水线开始时间
    pipeline_start = time.time()
    
    active_models = config.get_active_models()
    total_models = len(active_models)
    
    print(f"\n将训练 {total_models} 个模型 | Will train {total_models} models")
    
    for idx, (model_name, model_config) in enumerate(active_models.items(), 1):
        print(f"\n[{idx}/{total_models}] 训练模型 | Training model: {model_name}")
        
        # 记录实验开始
        log_experiment_start(
            pipeline_name='P1_baseline',
            model_name=model_name,
            train_samples=len(X_train),
            test_samples=len(X_test),
            features=X_train.shape[1]
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
        train_start = time.time()
        print(f"  训练中... | Training...")
        model.fit(X_train, y_train_model)
        train_time = time.time() - train_start
        print(f"  训练完成 | Training completed in {train_time:.2f}s")
        
        # 预测
        predict_start = time.time()
        print(f"  预测中... | Predicting...")
        y_pred = model.predict(X_test)
        predict_time = time.time() - predict_start
        print(f"  预测完成 | Prediction completed in {predict_time:.2f}s")
        
        # 如果是XGBoost，将预测结果转换回原始标签
        if model_name == 'xgboost':
            y_pred = np.array([inverse_mapping[p] for p in y_pred])
        
        # 计算指标
        print(f"  计算指标... | Calculating metrics...")
        metrics = calculate_metrics(y_test, y_pred, average=None)
        
        # 添加实验信息
        metrics['pipeline'] = 'P1_baseline'
        metrics['model'] = model_name
        metrics['train_time'] = train_time
        metrics['predict_time'] = predict_time
        metrics['total_time'] = train_time + predict_time
        metrics['train_samples'] = len(X_train)
        metrics['test_samples'] = len(X_test)
        metrics['num_features'] = X_train.shape[1]
        
        # 保存结果
        results_list.append(metrics)
        
        # 保存混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                            index=sorted(np.unique(y_test)),
                            columns=sorted(np.unique(y_test)))
        cm_df.to_csv(baseline_dir / f'confusion_matrix_{model_name}.csv')
        
        # 保存模型
        if config.OUTPUT['save_models']:
            save_model(model, baseline_dir, f'model_{model_name}.pkl')
        
        # 记录实验结束
        log_experiment_end('P1_baseline', model_name, metrics, train_time)
    
    # 计算流水线总时间
    pipeline_total_time = time.time() - pipeline_start
    
    # 保存所有结果
    results_df = pd.DataFrame(results_list)
    results_file = save_results(results_list, baseline_dir, 'results.csv')
    print(f"\n所有结果已保存 | All results saved: {results_file}")
    
    # 打印汇总表
    print("\n" + "=" * 60)
    print("基线实验结果汇总 | Baseline Results Summary")
    print(f"流水线总耗时 | Pipeline Total Time: {pipeline_total_time:.2f}s")
    print("=" * 60)
    summary = results_df[['model', 'accuracy', 'f1_macro', 'f1_weighted', 'train_time', 'predict_time']]
    print(summary.to_string(index=False))
    
    return results_list


def main():
    """
    主函数
    """
    print_section_header(
        "阶段2：基线实验 | Stage 2: Baseline Experiment",
        ""
    )
    
    # 创建输出目录
    output_dir = config.create_output_dir()
    print(f"输出目录 | Output directory: {output_dir}")
    
    # 检查是否有预处理后的数据
    # 这里简化处理，实际应从01_data_preprocessing.py的输出加载
    # 或者重新加载原始数据
    print("\n加载数据 | Loading data...")
    from utils import load_nsl_kdd_data, map_attack_categories, preprocess_features
    
    train_df, test_df = load_nsl_kdd_data()
    
    # 调试模式采样
    if config.DEBUG_MODE:
        print(f"[调试模式 | DEBUG MODE] 采样 {config.DEBUG_TRAIN_SIZE} 训练样本")
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
    
    print(f"训练集 | Training set: {X_train.shape}")
    print(f"测试集 | Test set: {X_test.shape}")
    
    # 运行基线实验
    results = run_baseline_experiment(X_train, y_train, X_test, y_test, output_dir)
    
    print_section_header(
        "基线实验完成 | Baseline Experiment Completed",
        ""
    )
    
    return results


if __name__ == "__main__":
    main()
