"""
断点继续模块 - 支持实验中断后恢复

功能:
1. 记录每组实验的完成状态
2. 支持从中断点自动恢复
3. 配置驱动 (读取config.py中的CHECKPOINT_CONFIG)
4. 保存/加载实验状态

使用方式:
    from checkpoint import CheckpointManager
    
    checkpoint = CheckpointManager(config)
    
    # 检查是否已完成
    if checkpoint.is_completed(experiment_id):
        print("跳过已完成实验")
        continue
    
    # 运行实验...
    
    # 标记为完成
    checkpoint.mark_completed(experiment_id, results)
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import config
from logger import get_logger


class CheckpointManager:
    """
    断点继续管理器
    
    配置驱动: 自动读取config.CHECKPOINT_CONFIG
    """
    
    def __init__(self, cfg=None):
        """
        初始化断点管理器
        
        参数:
            cfg: 配置模块，默认使用全局config
        """
        self.cfg = cfg or config
        self.logger = get_logger('checkpoint')
        
        # 读取配置
        cp_config = self.cfg.CHECKPOINT_CONFIG
        self.enabled = cp_config.get('enabled', True)
        self.auto_save = cp_config.get('auto_save', True)
        self.save_interval = cp_config.get('save_interval', 1)
        
        # 路径设置
        self.checkpoint_dir = Path(self.cfg.OUTPUT_DIR) / cp_config.get('checkpoint_dir', 'checkpoints')
        self.state_file = self.checkpoint_dir / cp_config.get('file_name', 'experiment_state.json')
        self.completed_file = self.checkpoint_dir / cp_config.get('completed_file', 'completed_experiments.csv')
        
        # 状态数据
        self.state = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'total_experiments': 0,
            'completed_experiments': 0,
            'experiments': {}  # experiment_id -> {status, timestamp, results_summary}
        }
        
        # 缓存计数器
        self._save_counter = 0
        
        # 初始化
        self._init_checkpoint_dir()
        self.load_state()
        
        if self.enabled:
            self.logger.info(f"断点继续已启用 | Checkpoint enabled: {self.checkpoint_dir}")
        else:
            self.logger.info("断点继续已禁用 | Checkpoint disabled")
    
    def _init_checkpoint_dir(self):
        """创建检查点目录"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_experiment_id(self, pipeline: str, model: str, 
                                fs_method: Optional[str] = None,
                                n_features: Optional[int] = None) -> str:
        """
        生成实验唯一标识符
        
        格式: P{pipeline}_{model}[_fs{fs_method}][_top{n_features}]
        
        示例:
            - P1_decision_tree
            - P2_random_forest
            - P3_decision_tree_filter_top10
            - P4_xgboost_wrapper_top20
            - P5_random_forest_filter_top30
        """
        parts = [pipeline, model]
        if fs_method:
            parts.append(f"fs{fs_method}")
        if n_features:
            parts.append(f"top{n_features}")
        
        return "_".join(parts)
    
    def load_state(self):
        """
        加载已保存的实验状态
        
        如果状态文件存在，读取并恢复；否则创建新状态
        """
        if not self.enabled:
            return
        
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                # 验证版本兼容性
                if loaded_state.get('version') == self.state['version']:
                    self.state = loaded_state
                    completed = self.state.get('completed_experiments', 0)
                    total = self.state.get('total_experiments', 0)
                    self.logger.info(
                        f"加载检查点 | Checkpoint loaded: {completed}/{total} 实验已完成"
                    )
                else:
                    self.logger.warning(
                        f"检查点版本不匹配，创建新状态 | Version mismatch, creating new state"
                    )
            except Exception as e:
                self.logger.error(f"加载检查点失败 | Failed to load checkpoint: {e}")
        else:
            self.logger.info("未找到检查点，从头开始 | No checkpoint found, starting fresh")
    
    def save_state(self, force: bool = False):
        """
        保存当前实验状态到文件
        
        参数:
            force: 是否强制保存（忽略auto_save和间隔）
        """
        if not self.enabled:
            return
        
        if not force and not self.auto_save:
            return
        
        self._save_counter += 1
        if not force and self._save_counter % self.save_interval != 0:
            return
        
        try:
            self.state['updated_at'] = datetime.now().isoformat()
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            
            self.logger.debug("检查点已保存 | Checkpoint saved")
        except Exception as e:
            self.logger.error(f"保存检查点失败 | Failed to save checkpoint: {e}")
    
    def is_completed(self, experiment_id: str) -> bool:
        """
        检查指定实验是否已完成
        
        参数:
            experiment_id: 实验标识符
            
        返回:
            bool: 是否已完成
        """
        if not self.enabled:
            return False
        
        exp = self.state['experiments'].get(experiment_id)
        if exp and exp.get('status') == 'completed':
            return True
        return False
    
    def mark_completed(self, experiment_id: str, 
                       results: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """
        标记实验为已完成
        
        参数:
            experiment_id: 实验标识符
            results: 实验结果摘要（可选）
            metadata: 额外元数据（可选）
        """
        if not self.enabled:
            return
        
        # 构建实验记录
        exp_record = {
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'results_summary': results or {},
            'metadata': metadata or {}
        }
        
        # 更新状态
        self.state['experiments'][experiment_id] = exp_record
        self.state['completed_experiments'] = len([
            e for e in self.state['experiments'].values() 
            if e.get('status') == 'completed'
        ])
        
        # 保存到CSV记录
        self._append_to_completed_csv(experiment_id, exp_record)
        
        # 自动保存状态
        self.save_state()
        
        self.logger.debug(f"实验标记完成 | Experiment marked: {experiment_id}")
    
    def _append_to_completed_csv(self, experiment_id: str, exp_record: Dict):
        """将完成的实验追加到CSV文件"""
        try:
            file_exists = self.completed_file.exists()
            
            with open(self.completed_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入表头
                if not file_exists:
                    writer.writerow([
                        'experiment_id', 'status', 'completed_at',
                        'f1_macro', 'f1_weighted', 'accuracy', 'notes'
                    ])
                
                # 提取关键结果
                results = exp_record.get('results_summary', {})
                writer.writerow([
                    experiment_id,
                    exp_record['status'],
                    exp_record['completed_at'],
                    results.get('f1_macro', ''),
                    results.get('f1_weighted', ''),
                    results.get('accuracy', ''),
                    exp_record['metadata'].get('notes', '')
                ])
        except Exception as e:
            self.logger.error(f"写入CSV失败 | CSV write failed: {e}")
    
    def mark_failed(self, experiment_id: str, error_msg: str):
        """
        标记实验为失败
        
        参数:
            experiment_id: 实验标识符
            error_msg: 错误信息
        """
        if not self.enabled:
            return
        
        self.state['experiments'][experiment_id] = {
            'status': 'failed',
            'failed_at': datetime.now().isoformat(),
            'error': error_msg
        }
        
        self.save_state()
        self.logger.error(f"实验失败 | Experiment failed: {experiment_id} - {error_msg}")
    
    def get_progress(self) -> Dict[str, Any]:
        """
        获取实验进度信息
        
        返回:
            dict: 包含total, completed, failed, remaining
        """
        experiments = self.state.get('experiments', {})
        completed = sum(1 for e in experiments.values() if e.get('status') == 'completed')
        failed = sum(1 for e in experiments.values() if e.get('status') == 'failed')
        total = self.state.get('total_experiments', 0)
        
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'remaining': total - completed - failed,
            'progress_pct': (completed / total * 100) if total > 0 else 0
        }
    
    def print_progress(self):
        """打印当前进度"""
        progress = self.get_progress()
        self.logger.info(
            f"实验进度 | Progress: {progress['completed']}/{progress['total']} "
            f"({progress['progress_pct']:.1f}%) "
            f"[失败: {progress['failed']} | 剩余: {progress['remaining']}]"
        )
    
    def set_total_experiments(self, total: int):
        """
        设置总实验数量（用于进度计算）
        
        参数:
            total: 总实验组数
        """
        self.state['total_experiments'] = total
        self.save_state(force=True)
    
    def get_completed_experiments(self) -> List[str]:
        """
        获取所有已完成的实验ID列表
        
        返回:
            list: 实验ID列表
        """
        return [
            exp_id for exp_id, exp in self.state['experiments'].items()
            if exp.get('status') == 'completed'
        ]
    
    def get_failed_experiments(self) -> List[str]:
        """
        获取所有失败的实验ID列表
        
        返回:
            list: 实验ID列表
        """
        return [
            exp_id for exp_id, exp in self.state['experiments'].items()
            if exp.get('status') == 'failed'
        ]
    
    def reset(self, confirm: bool = False):
        """
        重置所有状态（谨慎使用！）
        
        参数:
            confirm: 必须设为True才会执行
        """
        if not confirm:
            self.logger.warning("重置检查点需要confirm=True | Reset requires confirm=True")
            return
        
        self.state = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'total_experiments': 0,
            'completed_experiments': 0,
            'experiments': {}
        }
        
        # 删除文件
        if self.state_file.exists():
            self.state_file.unlink()
        if self.completed_file.exists():
            self.completed_file.unlink()
        
        self.logger.info("检查点已重置 | Checkpoint reset")
    
    def generate_id(self, pipeline: str, model: str,
                    fs_method: Optional[str] = None,
                    n_features: Optional[int] = None) -> str:
        """
        便捷方法：生成实验ID
        
        参数:
            pipeline: 流水线名称 (P1-P5)
            model: 模型名称
            fs_method: 特征选择方法 (filter/wrapper)
            n_features: 特征数量
            
        返回:
            str: 实验唯一标识符
        """
        return self._generate_experiment_id(pipeline, model, fs_method, n_features)


# ==================== 便捷函数 ====================

def create_checkpoint(cfg=None) -> CheckpointManager:
    """
    创建检查点管理器实例
    
    参数:
        cfg: 配置模块
        
    返回:
        CheckpointManager实例
    """
    return CheckpointManager(cfg)


def get_experiment_id(pipeline: str, model: str,
                     fs_method: Optional[str] = None,
                     n_features: Optional[int] = None) -> str:
    """
    生成实验ID的便捷函数
    
    示例:
        >>> get_experiment_id('P1', 'decision_tree')
        'P1_decision_tree'
        >>> get_experiment_id('P5', 'random_forest', 'filter', 20)
        'P5_random_forest_fsfilter_top20'
    """
    parts = [pipeline, model]
    if fs_method:
        parts.append(f"fs{fs_method}")
    if n_features:
        parts.append(f"top{n_features}")
    return "_".join(parts)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试断点继续模块 | Testing Checkpoint Module")
    print("=" * 60)
    
    # 创建测试实例
    cp = CheckpointManager()
    
    # 测试生成ID
    print("\n1. 测试实验ID生成:")
    test_ids = [
        cp.generate_id('P1', 'decision_tree'),
        cp.generate_id('P2', 'random_forest'),
        cp.generate_id('P3', 'xgboost', 'filter', 10),
        cp.generate_id('P4', 'decision_tree', 'wrapper', 20),
        cp.generate_id('P5', 'random_forest', 'filter', 30),
    ]
    for eid in test_ids:
        print(f"   {eid}")
    
    # 测试标记完成
    print("\n2. 测试标记完成:")
    cp.set_total_experiments(5)
    
    for i, eid in enumerate(test_ids):
        results = {
            'accuracy': 0.75 + i * 0.01,
            'f1_macro': 0.50 + i * 0.02,
            'f1_weighted': 0.65 + i * 0.01
        }
        cp.mark_completed(eid, results)
        print(f"   [OK] {eid}")
    
    # 测试检查完成状态
    print("\n3. 测试检查状态:")
    print(f"   P1_decision_tree 完成? {cp.is_completed('P1_decision_tree')}")
    print(f"   P1_random_forest 完成? {cp.is_completed('P1_random_forest')}")
    
    # 测试进度
    print("\n4. 测试进度报告:")
    cp.print_progress()
    
    # 测试重置
    print("\n5. 测试重置:")
    # cp.reset(confirm=True)  # 取消注释以测试重置
    print("   (跳过重置测试，保留检查点)")
    
    print("\n" + "=" * 60)
    print("测试完成 | Test completed")
    print("=" * 60)
