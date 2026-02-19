"""
PA-LOI 实验数据记录器 (Data Logger)

用于量化分析实验结果，支持参数调优和论文图表生成。
输出 CSV 格式，可直接用 pandas/Excel 分析。

使用方法:
    logger = PALOIDataLogger(scenario_id="S01", w_base=20.0, lambda_v=0.1)
    
    # 在每帧规划后调用
    logger.log_frame(
        ego_state=state,
        risk_sources=risk_sources,
        phantom_result=phantom_result,
        ...
    )
    
    # 实验结束时保存
    logger.save()
"""

import os
import csv
import time
import numpy as np
from datetime import datetime


class PALOIDataLogger:
    """
    PA-LOI 专属黑盒记录仪
    
    记录每帧的关键数据用于:
    1. 验证算法是否正常工作
    2. 参数调优 (Tuning)
    3. 生成论文图表
    """
    
    def __init__(self, scenario_id="default", w_base=10.0, lambda_v=0.1, 
                 output_dir="./logs"):
        """
        Args:
            scenario_id: 场景标识符
            w_base: 基础权重
            lambda_v: 速度系数
            output_dir: 日志输出目录
        """
        self.scenario_id = scenario_id
        self.w_base = w_base
        self.lambda_v = lambda_v
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # [Fix] 使用微秒级时间戳防止重名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        w_str = f"W{int(w_base)}"
        l_str = f"L{str(lambda_v).replace('.', '')}"
        self.filename = f"log_{scenario_id}_{timestamp}_{w_str}_{l_str}.csv"
        self.filepath = os.path.join(output_dir, self.filename)
        
        self.frame_count = 0
        self.start_time = time.time()
        self.min_dist_to_ghost = float('inf')
        self.collision_count = 0
        
        # CSV 列定义
        self.columns = [
            'Frame', 'Time', 'Ego_X', 'Ego_Y', 'Ego_Vel', 'Ego_Acc', 'Ego_Heading',
            'Risk_Source_Dist', 'D_Lat', 'D_Critical', 'D_Outer',
            'Risk_Cost_Raw', 'Vel_Factor', 'Risk_Cost_Total',
            'TTA_Ego', 'TTA_Human', 'V_Required',
            'Phantom_State', 'Is_Phantom_Active', 'Phantom_Virtual_Dist',
            'Min_Dist_To_Ghost', 'Is_Collision',
            'Ctrl_Acc', 'Ctrl_Steer'
        ]
        
        # [Fix] 立即打开文件并写入 Header
        self.file_handle = open(self.filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.columns)
        self.writer.writeheader()
        self.file_handle.flush()
        os.fsync(self.file_handle.fileno())
        
        print(f"[PA-LOI Logger] Initialized (Stream Mode): {self.filepath}")
    
    def log_frame(self, ego_state, risk_sources=None, phantom_result=None,
                  d_critical=None, d_outer=None, ctrl=None, is_collision=False):
        """记录单帧数据 (立即写入磁盘)"""
        self.frame_count += 1
        current_time = time.time() - self.start_time
        
        # ... (解析逻辑保持不变，为了简化diff，这里不再重复解析代码，假设下方row构建使用了解析后的变量) ...
        # [注意] 由于 replace_file_content 无法只替换函数头尾保留中间，我必须把中间的解析逻辑也写上。
        # 为了避免超长，我将尽量保持原样。
        
        # 解析 ego_state
        ego_x = ego_state[0] if len(ego_state) > 0 else 0.0
        ego_y = ego_state[1] if len(ego_state) > 1 else 0.0
        ego_vel = ego_state[2] if len(ego_state) > 2 else 0.0
        ego_heading = ego_state[3] if len(ego_state) > 3 else 0.0
        ego_acc = ego_state[4] if len(ego_state) > 4 else 0.0
        
        ctrl_acc = ctrl[0] if ctrl is not None and len(ctrl) > 0 else 0.0
        ctrl_steer = ctrl[1] if ctrl is not None and len(ctrl) > 1 else 0.0
        
        risk_source_dist = float('inf')
        d_lat = float('inf')
        risk_cost_raw = 0.0
        vel_factor = 1.0
        risk_cost_total = 0.0
        ghost_lateral = 0.0
        ghost_longitudinal = 0.0
        
        if risk_sources and len(risk_sources) > 0:
            closest_risk = risk_sources[0]
            risk_pos = closest_risk['pos']
            if hasattr(risk_pos, 'cpu'): risk_pos = risk_pos.cpu().numpy()
            dx = ego_x - risk_pos[0]
            dy = ego_y - risk_pos[1]
            risk_source_dist = np.sqrt(dx**2 + dy**2)
            ghost_lateral = closest_risk.get('ghost_lateral', 1.5)
            ghost_longitudinal = closest_risk.get('ghost_longitudinal', 0.0)
            d_lat = ghost_lateral
            
            clearance = max(d_lat - 1.0, 0.0)
            exp_arg = np.clip(2.0 * (clearance - ghost_lateral), -10, 10)
            sigmoid_val = 1.0 / (1.0 + np.exp(exp_arg))
            w_base = closest_risk.get('weight', self.w_base)
            risk_cost_raw = w_base * sigmoid_val
            
            # [Fix] 适配 Hinge Loss 的显示 (Vel Factor 不再是 1+kv^2)
            v_safe = closest_risk.get('v_safe', 0.0)
            excess = max(0.0, ego_vel - v_safe)
            vel_factor = excess * excess # Hinge Loss Squared
            risk_cost_total = risk_cost_raw * vel_factor
            
            if risk_source_dist < self.min_dist_to_ghost:
                self.min_dist_to_ghost = risk_source_dist
        
        tta_ego = float('inf')
        tta_human = float('inf')
        v_required = 0.0
        phantom_state = 0
        is_phantom_active = 0
        phantom_virtual_dist = 0.0
        
        if phantom_result is not None:
            tta_ego = phantom_result.get('tta_ego', float('inf'))
            tta_human = phantom_result.get('tta_human', float('inf'))
            v_required = phantom_result.get('v_required', 0.0)
            state_str = phantom_result.get('state', 'OBSERVE')
            phantom_state = {'OBSERVE': 0, 'BRAKE': 1, 'PASS': 2}.get(state_str, 0)
            is_phantom_active = 1 if phantom_result.get('inject_phantom', False) else 0
            phantom_virtual_dist = ghost_longitudinal
        
        if is_collision:
            self.collision_count += 1
        
        row = {
            'Frame': self.frame_count,
            'Time': round(current_time, 3),
            'Ego_X': round(ego_x, 3), 'Ego_Y': round(ego_y, 3),
            'Ego_Vel': round(ego_vel, 3), 'Ego_Acc': round(ego_acc, 3), 'Ego_Heading': round(ego_heading, 4),
            'Risk_Source_Dist': round(risk_source_dist, 3) if risk_source_dist != float('inf') else -1,
            'D_Lat': round(d_lat, 3) if d_lat != float('inf') else -1,
            'D_Critical': round(d_critical, 3) if d_critical is not None else -1,
            'D_Outer': round(d_outer, 3) if d_outer is not None else -1,
            'Risk_Cost_Raw': round(risk_cost_raw, 4),
            'Vel_Factor': round(vel_factor, 4),
            'Risk_Cost_Total': round(risk_cost_total, 4),
            'TTA_Ego': round(tta_ego, 3) if tta_ego != float('inf') else -1,
            'TTA_Human': round(tta_human, 3) if tta_human != float('inf') else -1,
            'V_Required': round(v_required, 3),
            'Phantom_State': phantom_state,
            'Is_Phantom_Active': is_phantom_active,
            'Phantom_Virtual_Dist': round(phantom_virtual_dist, 3),
            'Min_Dist_To_Ghost': round(self.min_dist_to_ghost, 3) if self.min_dist_to_ghost != float('inf') else -1,
            'Is_Collision': 1 if is_collision else 0,
            'Ctrl_Acc': round(ctrl_acc, 4), 'Ctrl_Steer': round(ctrl_steer, 4)
        }
        
        # [Fix] 立即写入并 Flash
        if self.writer:
            self.writer.writerow(row)
            self.file_handle.flush()
            os.fsync(self.file_handle.fileno())
            
        if self.frame_count % 100 == 0:
            print(f"[PA-LOI Logger] Frame {self.frame_count}: v={ego_vel:.1f}m/s, cost={risk_cost_total:.2f}")

    def save(self):
        """关闭文件并打印统计摘要"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"[PA-LOI Logger] Experiment Complete!")
        print(f"  Output File: {self.filepath}")
        print(f"  Total Frames: {self.frame_count}")
        print(f"  Collisions: {self.collision_count}")
        print(f"{'='*60}\n")
        return self.filepath
    
    def get_summary(self):
        # 简化版：由于不再存 self.data，无法计算均值统计，但这不影响 log 文件本身
        return {'total_frames': self.frame_count}


# ============= 快速分析工具 =============

def load_experiment_log(filepath):
    """
    加载实验日志用于分析
    
    Args:
        filepath: CSV 文件路径
    
    Returns:
        pandas DataFrame (如果 pandas 可用) 或 list of dicts
    """
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        print(f"[PA-LOI] Loaded {len(df)} frames from {filepath}")
        return df
    except ImportError:
        # 如果没有 pandas，返回原始数据
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        print(f"[PA-LOI] Loaded {len(data)} frames (pandas not available)")
        return data


def plot_experiment_results(filepath):
    """
    绘制实验结果图表 (用于论文)
    
    生成三个子图:
    1. 速度随时间变化
    2. 横向距离和 Cost 随时间变化
    3. 幻影状态随时间变化
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PA-LOI] matplotlib or pandas not available for plotting")
        return
    
    df = pd.read_csv(filepath)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 子图1: 速度
    ax1 = axes[0]
    ax1.plot(df['Time'], df['Ego_Vel'], 'b-', linewidth=2, label='Ego Velocity')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('PA-LOI Experiment Results')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 距离 和 Cost
    ax2 = axes[1]
    ax2.plot(df['Time'], df['D_Lat'], 'g-', linewidth=2, label='Lateral Distance')
    ax2.plot(df['Time'], df['D_Critical'], 'r--', linewidth=1, label='D_Critical')
    ax2.set_ylabel('Distance (m)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    ax2b = ax2.twinx()
    ax2b.plot(df['Time'], df['Risk_Cost_Total'], 'orange', linewidth=2, label='Risk Cost')
    ax2b.set_ylabel('Cost', color='orange')
    ax2b.legend(loc='upper right')
    
    # 子图3: 幻影状态
    ax3 = axes[2]
    ax3.fill_between(df['Time'], df['Is_Phantom_Active'], alpha=0.3, color='red', label='Phantom Active')
    ax3.plot(df['Time'], df['Phantom_State'], 'k-', linewidth=2, label='State (0:Obs, 1:Brake, 2:Pass)')
    ax3.set_ylabel('Phantom State')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = filepath.replace('.csv', '_plot.png')
    plt.savefig(output_path, dpi=150)
    print(f"[PA-LOI] Plot saved to {output_path}")
    plt.close()
    
    return output_path
