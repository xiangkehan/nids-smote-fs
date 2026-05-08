"""
主运行脚本 - 整合所有实验流水线 (支持断点继续)

功能:
1. 按顺序运行所有实验（P1-P5）
2. 支持断点继续（配置驱动）
3. 生成综合对比结果
4. 保存完整的实验日志

使用方式:
    python run_experiments.py --all          # 运行所有实验
    python run_experiments.py --pipeline P1  # 仅运行P1基线
    python run_experiments.py --resume       # 从断点继续
    python run_experiments.py --reset        # 重置检查点
"""

import sys
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent))

import config
from utils import print_section_header, load_nsl_kdd_data, map_attack_categories, preprocess_features
from checkpoint import CheckpointManager, get_experiment_id

# 导入各个实验模块的核心函数
from baseline import run_baseline_experiment
from smote_only import run_smote_only_experiment
from feature_selection import run_feature_selection_experiment
from joint_pipeline_smote_fs import run_joint_smote_fs_experiment
from joint_pipeline_fs_smote import run_joint_fs_smote_experiment


def load_and_prepare_data():
    """加载并预处理数据"""
    print("\n加载数据 | Loading data...")
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
    
    return X_train, y_train, X_test, y_test


def calculate_total_experiments():
    """计算总实验组数"""
    active_models = config.get_active_models()
    active_fs = config.get_active_fs_methods()
    pipelines = config.get_active_pipelines()
    
    total = 0
    
    # P1: 基线 (模型数)
    if pipelines.get('P1_baseline', False):
        total += len(active_models)
    
    # P2: 仅SMOTE (模型数)
    if pipelines.get('P2_smote_only', False):
        total += len(active_models)
    
    # P3: 仅特征选择 (模型 × FS方法 × 阈值数)
    if pipelines.get('P3_feature_selection_only', False):
        for fs_config in active_fs.values():
            total += len(active_models) * len(fs_config['thresholds'])
    
    # P4: SMOTE→FS (模型 × FS方法 × 阈值数)
    if pipelines.get('P4_smote_then_fs', False):
        for fs_config in active_fs.values():
            total += len(active_models) * len(fs_config['thresholds'])
    
    # P5: FS→SMOTE (模型 × FS方法 × 阈值数)
    if pipelines.get('P5_fs_then_smote', False):
        for fs_config in active_fs.values():
            total += len(active_models) * len(fs_config['thresholds'])
    
    return total


def run_all_experiments(checkpoint=None, resume=False):
    """
    运行所有实验流水线（支持断点继续）
    
    参数:
        checkpoint: CheckpointManager实例
        resume: 是否从断点恢复
    """
    start_time = time.time()
    
    # 初始化检查点
    if checkpoint is None:
        checkpoint = CheckpointManager()
    
    # 计算总实验数
    total_exps = calculate_total_experiments()
    checkpoint.set_total_experiments(total_exps)
    
    print("\n" + "="*70)
    print("开始运行所有实验 | Starting all experiments")
    if resume:
        print("模式: 断点继续 | Mode: Resume from checkpoint")
    print(f"总实验组数 | Total experiments: {total_exps}")
    checkpoint.print_progress()
    print("="*70)
    
    # 加载数据（所有流水线共享）
    X_train, y_train, X_test, y_test = load_and_prepare_data()
    output_dir = config.create_output_dir()
    print(f"输出目录 | Output directory: {output_dir}")
    
    active_models = list(config.get_active_models().keys())
    active_pipelines = config.get_active_pipelines()
    active_fs = config.get_active_fs_methods()
    
    # 计数器
    exp_counter = 0
    skipped = 0
    completed = 0
    failed = 0
    
    # ==================== P1: 基线 ====================
    if active_pipelines.get('P1_baseline', False):
        print_section_header("P1-基线实验 | P1-Baseline", "")
        
        for model_name in active_models:
            exp_id = get_experiment_id('P1', model_name)
            exp_counter += 1
            
            # 检查是否已完成
            if resume and checkpoint.is_completed(exp_id):
                print(f"[{exp_counter}/{total_exps}] 跳过已完成: {exp_id}")
                skipped += 1
                continue
            
            print(f"\n[{exp_counter}/{total_exps}] 运行: {exp_id}")
            try:
                results = run_baseline_experiment(X_train, y_train, X_test, y_test, output_dir)
                # 标记完成
                for r in results:
                    if r['model'] == model_name:
                        checkpoint.mark_completed(exp_id, {
                            'accuracy': r.get('accuracy'),
                            'f1_macro': r.get('f1_macro'),
                            'f1_weighted': r.get('f1_weighted')
                        })
                        break
                completed += 1
            except Exception as e:
                checkpoint.mark_failed(exp_id, str(e))
                failed += 1
                print(f"错误: {e}")
    
    # ==================== P2: 仅SMOTE ====================
    if active_pipelines.get('P2_smote_only', False):
        print_section_header("P2-仅SMOTE | P2-SMOTE Only", "")
        
        for model_name in active_models:
            exp_id = get_experiment_id('P2', model_name)
            exp_counter += 1
            
            if resume and checkpoint.is_completed(exp_id):
                print(f"[{exp_counter}/{total_exps}] 跳过已完成: {exp_id}")
                skipped += 1
                continue
            
            print(f"\n[{exp_counter}/{total_exps}] 运行: {exp_id}")
            try:
                results = run_smote_only_experiment(X_train, y_train, X_test, y_test, output_dir)
                for r in results:
                    if r['model'] == model_name:
                        checkpoint.mark_completed(exp_id, {
                            'accuracy': r.get('accuracy'),
                            'f1_macro': r.get('f1_macro'),
                            'f1_weighted': r.get('f1_weighted')
                        })
                        break
                completed += 1
            except Exception as e:
                checkpoint.mark_failed(exp_id, str(e))
                failed += 1
                print(f"错误: {e}")
    
    # ==================== P3: 仅特征选择 ====================
    if active_pipelines.get('P3_feature_selection_only', False):
        print_section_header("P3-仅特征选择 | P3-Feature Selection Only", "")
        
        for fs_name, fs_config in active_fs.items():
            for k in fs_config['thresholds']:
                for model_name in active_models:
                    exp_id = get_experiment_id('P3', model_name, fs_name, k)
                    exp_counter += 1
                    
                    if resume and checkpoint.is_completed(exp_id):
                        print(f"[{exp_counter}/{total_exps}] 跳过已完成: {exp_id}")
                        skipped += 1
                        continue
                    
                    print(f"\n[{exp_counter}/{total_exps}] 运行: {exp_id}")
                    try:
                        results = run_feature_selection_experiment(
                            X_train, y_train, X_test, y_test, output_dir,
                            fs_method=fs_name, k=k
                        )
                        for r in results:
                            if r['model'] == model_name:
                                checkpoint.mark_completed(exp_id, {
                                    'accuracy': r.get('accuracy'),
                                    'f1_macro': r.get('f1_macro'),
                                    'f1_weighted': r.get('f1_weighted')
                                })
                                break
                        completed += 1
                    except Exception as e:
                        checkpoint.mark_failed(exp_id, str(e))
                        failed += 1
                        print(f"错误: {e}")
    
    # ==================== P4: SMOTE→FS ====================
    if active_pipelines.get('P4_smote_then_fs', False):
        print_section_header("P4-SMOTE→FS | P4-SMOTE then FS", "")
        
        for fs_name, fs_config in active_fs.items():
            for k in fs_config['thresholds']:
                for model_name in active_models:
                    exp_id = get_experiment_id('P4', model_name, fs_name, k)
                    exp_counter += 1
                    
                    if resume and checkpoint.is_completed(exp_id):
                        print(f"[{exp_counter}/{total_exps}] 跳过已完成: {exp_id}")
                        skipped += 1
                        continue
                    
                    print(f"\n[{exp_counter}/{total_exps}] 运行: {exp_id}")
                    try:
                        results = run_joint_smote_fs_experiment(
                            X_train, y_train, X_test, y_test, output_dir,
                            fs_method=fs_name, k=k
                        )
                        for r in results:
                            if r['model'] == model_name:
                                checkpoint.mark_completed(exp_id, {
                                    'accuracy': r.get('accuracy'),
                                    'f1_macro': r.get('f1_macro'),
                                    'f1_weighted': r.get('f1_weighted')
                                })
                                break
                        completed += 1
                    except Exception as e:
                        checkpoint.mark_failed(exp_id, str(e))
                        failed += 1
                        print(f"错误: {e}")
    
    # ==================== P5: FS→SMOTE (核心) ====================
    if active_pipelines.get('P5_fs_then_smote', False):
        print_section_header("P5-FS→SMOTE (核心) | P5-FS then SMOTE (Core)", "")
        
        for fs_name, fs_config in active_fs.items():
            for k in fs_config['thresholds']:
                for model_name in active_models:
                    exp_id = get_experiment_id('P5', model_name, fs_name, k)
                    exp_counter += 1
                    
                    if resume and checkpoint.is_completed(exp_id):
                        print(f"[{exp_counter}/{total_exps}] 跳过已完成: {exp_id}")
                        skipped += 1
                        continue
                    
                    print(f"\n[{exp_counter}/{total_exps}] 运行: {exp_id}")
                    try:
                        results = run_joint_fs_smote_experiment(
                            X_train, y_train, X_test, y_test, output_dir,
                            fs_method=fs_name, k=k
                        )
                        for r in results:
                            if r['model'] == model_name:
                                checkpoint.mark_completed(exp_id, {
                                    'accuracy': r.get('accuracy'),
                                    'f1_macro': r.get('f1_macro'),
                                    'f1_weighted': r.get('f1_weighted')
                                })
                                break
                        completed += 1
                    except Exception as e:
                        checkpoint.mark_failed(exp_id, str(e))
                        failed += 1
                        print(f"错误: {e}")
    
    # 最终总结
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("所有实验完成！| All experiments completed!")
    print(f"总耗时 | Total time: {total_time/60:.2f} 分钟")
    print(f"完成 | Completed: {completed}")
    print(f"跳过 | Skipped: {skipped}")
    print(f"失败 | Failed: {failed}")
    checkpoint.print_progress()
    print("="*70)
    
    # 强制保存最终状态
    checkpoint.save_state(force=True)


def run_specific_pipeline(pipeline_name, checkpoint=None, resume=False):
    """
    运行指定的流水线
    
    参数:
        pipeline_name: 流水线名称 (P1, P2, P3, P4, P5)
        checkpoint: CheckpointManager实例
        resume: 是否从断点恢复
    """
    if checkpoint is None:
        checkpoint = CheckpointManager()
    
    X_train, y_train, X_test, y_test = load_and_prepare_data()
    output_dir = config.create_output_dir()
    
    active_models = list(config.get_active_models().keys())
    active_fs = config.get_active_fs_methods()
    
    pipeline_map = {
        'P1': ('P1-基线', lambda: [
            run_baseline_experiment(X_train, y_train, X_test, y_test, output_dir)
        ]),
        'P2': ('P2-仅SMOTE', lambda: [
            run_smote_only_experiment(X_train, y_train, X_test, y_test, output_dir)
        ]),
    }
    
    if pipeline_name not in pipeline_map:
        print(f"错误: 不支持的流水线 '{pipeline_name}'")
        print("支持: P1, P2, P3, P4, P5")
        return
    
    name, func = pipeline_map[pipeline_name]
    print(f"\n运行: {name}")
    func()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SMOTE与特征选择联合流水线实验 (支持断点继续)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_experiments.py --all              # 运行所有实验
  python run_experiments.py --pipeline P1      # 仅运行基线
  python run_experiments.py --pipeline P5      # 仅运行核心实验
  python run_experiments.py --resume           # 从断点继续
  python run_experiments.py --reset            # 重置检查点
  python run_experiments.py --status           # 查看当前进度
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='运行所有实验')
    parser.add_argument('--pipeline', type=str,
                       choices=['P1', 'P2', 'P3', 'P4', 'P5'],
                       help='运行指定的流水线')
    parser.add_argument('--resume', action='store_true',
                       help='从断点继续（跳过已完成实验）')
    parser.add_argument('--reset', action='store_true',
                       help='重置检查点（删除所有进度）')
    parser.add_argument('--status', action='store_true',
                       help='查看当前实验进度')
    parser.add_argument('--config', action='store_true',
                       help='显示当前配置')
    
    args = parser.parse_args()
    
    # 显示配置
    if args.config:
        config.print_config()
        return
    
    # 查看进度
    if args.status:
        checkpoint = CheckpointManager()
        checkpoint.print_progress()
        completed = checkpoint.get_completed_experiments()
        if completed:
            print(f"\n已完成实验 | Completed experiments:")
            for exp_id in completed[:10]:  # 显示前10个
                print(f"  [OK] {exp_id}")
            if len(completed) > 10:
                print(f"  ... 等共 {len(completed)} 个")
        return
    
    # 重置检查点
    if args.reset:
        checkpoint = CheckpointManager()
        checkpoint.reset(confirm=True)
        print("检查点已重置 | Checkpoint has been reset")
        return
    
    # 运行实验
    checkpoint = CheckpointManager()
    
    if args.all:
        run_all_experiments(checkpoint, resume=args.resume)
    elif args.pipeline:
        run_specific_pipeline(args.pipeline, checkpoint, resume=args.resume)
    elif args.resume:
        print("使用 --resume 配合 --all 或 --pipeline")
        print("例如: python run_experiments.py --all --resume")
    else:
        # 默认运行所有
        print("未指定参数，默认运行所有实验")
        print("使用 --help 查看所有选项")
        run_all_experiments(checkpoint, resume=False)


if __name__ == "__main__":
    main()
