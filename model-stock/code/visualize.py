import matplotlib.pyplot as plt
import numpy as np
import json
import os
from model_stock_lora import load_lora_weights, compute_angle_lora, compute_ratio

def visualize_angles_and_ratios(angles_dict, ratios_dict, save_path="../stock_test_output/"):
    """각도와 비율을 시각화합니다."""
    
    # 데이터 준비
    keys = list(angles_dict.keys())
    angles = [float(angles_dict[key]) for key in keys]
    ratios = [float(ratios_dict[key]) for key in keys]
    
    # 레이어 타입별 색상 분류
    colors = []
    for key in keys:
        if 'lora_A' in key:
            colors.append('blue')
        elif 'lora_B' in key:
            colors.append('red')
        elif 'q_proj' in key or 'k_proj' in key or 'v_proj' in key or 'o_proj' in key:
            colors.append('green')
        elif 'gate_proj' in key or 'up_proj' in key or 'down_proj' in key:
            colors.append('orange')
        else:
            colors.append('gray')
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 각도 시각화
    x = list(range(len(angles)))
    ax1.scatter(x, angles, c=colors, alpha=0.7)
    ax1.set_xlabel('Parameter Index')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Angles between Fine-tuned LoRA Adapters')
    ax1.grid(True, alpha=0.3)
    
    # 범례 추가
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='LoRA A'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='LoRA B'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Attention'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='MLP'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Others')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # 비율 시각화
    ax2.scatter(x, ratios, c=colors, alpha=0.7)
    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Interpolation Ratio')
    ax2.set_title('Model Stock Interpolation Ratios')
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # 저장
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'model_stock_visualization.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_file}")
    
    plt.show()
    
    return fig

def plot_angle_distribution(angles_dict, save_path="../stock_test_output/"):
    """각도 분포를 히스토그램으로 시각화합니다."""
    
    angles = [float(angle) for angle in angles_dict.values()]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(angles, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Angles between LoRA Adapters')
    ax.grid(True, alpha=0.3)
    
    # 통계 정보 추가
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    ax.axvline(mean_angle, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_angle:.2f}°')
    ax.axvline(mean_angle + std_angle, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_angle:.2f}°')
    ax.axvline(mean_angle - std_angle, color='orange', linestyle='--', alpha=0.7)
    
    ax.legend()
    
    # 저장
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'angle_distribution.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Angle distribution saved to {save_file}")
    
    plt.show()
    
    return fig

def create_summary_report(angles_dict, ratios_dict, save_path="../stock_test_output/"):
    """결과 요약 보고서를 생성합니다."""
    
    angles = [float(angle) for angle in angles_dict.values()]
    ratios = [float(ratio) for ratio in ratios_dict.values()]
    
    # 통계 계산
    angle_stats = {
        'mean': np.mean(angles),
        'std': np.std(angles),
        'min': np.min(angles),
        'max': np.max(angles),
        'median': np.median(angles)
    }
    
    ratio_stats = {
        'mean': np.mean(ratios),
        'std': np.std(ratios),
        'min': np.min(ratios),
        'max': np.max(ratios),
        'median': np.median(ratios)
    }
    
    # 레이어별 분석
    layer_analysis = {}
    for key, angle in angles_dict.items():
        layer_type = 'other'
        if 'lora_A' in key:
            layer_type = 'lora_A'
        elif 'lora_B' in key:
            layer_type = 'lora_B'
        elif any(proj in key for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            layer_type = 'attention'
        elif any(proj in key for proj in ['gate_proj', 'up_proj', 'down_proj']):
            layer_type = 'mlp'
        
        if layer_type not in layer_analysis:
            layer_analysis[layer_type] = {'angles': [], 'ratios': []}
        
        layer_analysis[layer_type]['angles'].append(float(angle))
        layer_analysis[layer_type]['ratios'].append(float(ratios_dict[key]))
    
    # 보고서 생성
    report = {
        'summary': {
            'total_parameters': len(angles_dict),
            'angle_statistics': angle_stats,
            'ratio_statistics': ratio_stats
        },
        'layer_analysis': {}
    }
    
    for layer_type, data in layer_analysis.items():
        report['layer_analysis'][layer_type] = {
            'count': len(data['angles']),
            'angle_mean': np.mean(data['angles']),
            'angle_std': np.std(data['angles']),
            'ratio_mean': np.mean(data['ratios']),
            'ratio_std': np.std(data['ratios'])
        }
    
    # JSON으로 저장
    os.makedirs(save_path, exist_ok=True)
    report_file = os.path.join(save_path, 'model_stock_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 텍스트 보고서도 생성
    text_report_file = os.path.join(save_path, 'model_stock_report.txt')
    with open(text_report_file, 'w') as f:
        f.write("Model Stock Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Parameters Analyzed: {len(angles_dict)}\n\n")
        
        f.write("Angle Statistics:\n")
        f.write(f"  Mean: {angle_stats['mean']:.2f}°\n")
        f.write(f"  Std:  {angle_stats['std']:.2f}°\n")
        f.write(f"  Min:  {angle_stats['min']:.2f}°\n")
        f.write(f"  Max:  {angle_stats['max']:.2f}°\n")
        f.write(f"  Median: {angle_stats['median']:.2f}°\n\n")
        
        f.write("Ratio Statistics:\n")
        f.write(f"  Mean: {ratio_stats['mean']:.4f}\n")
        f.write(f"  Std:  {ratio_stats['std']:.4f}\n")
        f.write(f"  Min:  {ratio_stats['min']:.4f}\n")
        f.write(f"  Max:  {ratio_stats['max']:.4f}\n")
        f.write(f"  Median: {ratio_stats['median']:.4f}\n\n")
        
        f.write("Layer-wise Analysis:\n")
        for layer_type, stats in report['layer_analysis'].items():
            f.write(f"  {layer_type.upper()}:\n")
            f.write(f"    Count: {stats['count']}\n")
            f.write(f"    Angle Mean: {stats['angle_mean']:.2f}°\n")
            f.write(f"    Ratio Mean: {stats['ratio_mean']:.4f}\n\n")
    
    print(f"Report saved to {report_file} and {text_report_file}")
    
    return report

def main():
    """메인 시각화 함수"""
    
    # 경로 설정
    finetuned_path_1 = '../stock_test_output/first_model/'
    finetuned_path_2 = '../stock_test_output/second_model/'
    save_path = '../stock_test_output/'
    
    try:
        # LoRA 가중치 로드
        print("Loading LoRA weights for visualization...")
        lora_weights_1 = load_lora_weights(finetuned_path_1)
        lora_weights_2 = load_lora_weights(finetuned_path_2)
        
        # 각도 계산
        print("Computing angles...")
        angles_dict = compute_angle_lora(lora_weights_1, lora_weights_2)
        
        if not angles_dict:
            print("No angles computed. Check your model files.")
            return
        
        # 비율 계산
        print("Computing ratios...")
        ratios_dict = compute_ratio(angles_dict, k=2)
        
        # 시각화
        print("Creating visualizations...")
        visualize_angles_and_ratios(angles_dict, ratios_dict, save_path)
        plot_angle_distribution(angles_dict, save_path)
        
        # 보고서 생성
        print("Generating summary report...")
        create_summary_report(angles_dict, ratios_dict, save_path)
        
        print("All visualizations and reports completed!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()