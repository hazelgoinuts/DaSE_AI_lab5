import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

class MetricsPlotter:
    def __init__(self, save_dir):
        """
        初始化MetricsPlotter
        Args:
            save_dir: 基础保存目录
        """
        self.base_dir = Path(save_dir)
        # 创建时间戳子文件夹
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = self.base_dir / timestamp
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # 记录最佳值
        self.best_metrics = {
            'train_loss': {'value': float('inf'), 'epoch': 0},
            'val_loss': {'value': float('inf'), 'epoch': 0},
            'train_acc': {'value': 0, 'epoch': 0},
            'val_acc': {'value': 0, 'epoch': 0}
        }
        
        # 添加新的性能指标
        self.performance_metrics = {
            'model_size': 0,  # 模型参数量
            'inference_time': 0,  # 平均推理时间
            'peak_memory': 0,  # 峰值内存占用
            'training_time': 0  # 总训练时间
        }
    
    def update(self, train_loss, val_loss, train_acc, val_acc):
        """更新指标"""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_acc'].append(val_acc)
        
        # 更新最佳值
        current_epoch = len(self.metrics['train_loss'])
        metrics_to_check = {
            'train_loss': (train_loss, min),
            'val_loss': (val_loss, min),
            'train_acc': (train_acc, max),
            'val_acc': (val_acc, max)
        }
        
        for metric_name, (value, comp) in metrics_to_check.items():
            if comp == min:
                if value < self.best_metrics[metric_name]['value']:
                    self.best_metrics[metric_name] = {'value': value, 'epoch': current_epoch-1}
            else:
                if value > self.best_metrics[metric_name]['value']:
                    self.best_metrics[metric_name] = {'value': value, 'epoch': current_epoch-1}
    
    def update_performance_metrics(self, model):
        # 计算模型大小
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        self.performance_metrics['model_size'] = (param_size + buffer_size) / 1024**2  # 转换为MB
    
    def save_config(self, config_dict):
        """保存配置文件"""
        config_path = self.save_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def plot(self):
        """绘制损失曲线和准确率曲线"""
        plt.figure(figsize=(15, 6))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        epochs = range(len(self.metrics['train_loss']))
        
        # 绘制主曲线
        plt.plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss', marker='o')
        plt.plot(epochs, self.metrics['val_loss'], 'r-', label='Val Loss', marker='s')
        
        # 标注数值
        for i, (train_loss, val_loss) in enumerate(zip(self.metrics['train_loss'], self.metrics['val_loss'])):
            plt.annotate(f'{train_loss:.3f}', (i, train_loss), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(f'{val_loss:.3f}', (i, val_loss), textcoords="offset points", xytext=(0,-15), ha='center')
        
        # 标注最佳点
        best_train_loss = self.best_metrics['train_loss']
        best_val_loss = self.best_metrics['val_loss']
        plt.plot(best_train_loss['epoch'], best_train_loss['value'], 'bo', markersize=10, fillstyle='none')
        plt.plot(best_val_loss['epoch'], best_val_loss['value'], 'ro', markersize=10, fillstyle='none')
        
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.metrics['train_acc'], 'b-', label='Train Acc', marker='o')
        plt.plot(epochs, self.metrics['val_acc'], 'r-', label='Val Acc', marker='s')
        
        # 标注数值
        for i, (train_acc, val_acc) in enumerate(zip(self.metrics['train_acc'], self.metrics['val_acc'])):
            plt.annotate(f'{train_acc:.3f}', (i, train_acc), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(f'{val_acc:.3f}', (i, val_acc), textcoords="offset points", xytext=(0,-15), ha='center')
        
        # 标注最佳点
        best_train_acc = self.best_metrics['train_acc']
        best_val_acc = self.best_metrics['val_acc']
        plt.plot(best_train_acc['epoch'], best_train_acc['value'], 'bo', markersize=10, fillstyle='none')
        plt.plot(best_val_acc['epoch'], best_val_acc['value'], 'ro', markersize=10, fillstyle='none')
        
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_performance_report(self):
        report = {
            'model_config': self.config_dict['model_config'],
            'performance_metrics': self.performance_metrics,
            'best_metrics': self.best_metrics
        }
        
        report_path = self.save_dir / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4) 