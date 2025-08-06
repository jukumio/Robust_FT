import matplotlib.pyplot as plt
import numpy as np
import json
import os
import torch
from safetensors.torch import load_file
from model_stock_lora import load_base_model_weights, load_lora_weights, create_lora_mapping, compute_lora_delta, reconstruct_full_weights, compute_angles_model_stock, compute_interpolation_ratios

def extract_folder_name(path):
    """경로에서 폴더 이름을 추출합니다."""
    return os.path.basename(os.path.normpath(path))

def visualize_angles_and_ratios(angles_dict, ratios_dict, save_path, model1_path=None, model2_path=None, suffix=""):
    """각도와 비율을 시각화합니다."""
    # 데이터 준비
    keys = list(angles_dict.keys())
    angles = [float(angles_dict[key]) for key in keys]
    ratios = [float(ratios_dict[key]) for key in keys]

    # 레이어 타입별 색상 분류
    colors = []
    for key in keys:
        if 'q_proj' in key or 'k_proj' in key or 'v_proj' in key or 'o_proj' in key:
            colors.append('green')  # attention
        elif 'gate_proj' in key or 'up_proj' in key or 'down_proj' in key:
            colors.append('orange')  # mlp
        else:
            colors.append('gray')

    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 제목 설정
    title_suffix = suffix if suffix else ""
    if model1_path and model2_path:
        model1_name = extract_folder_name(model1_path)
        model2_name = extract_folder_name(model2_path)
        angles_title = f'Angles between {model1_name} and {model2_name} {title_suffix}'
        ratios_title = f'Model Stock Interpolation Ratios between {model1_name} and {model2_name} {title_suffix}'
    else:
        angles_title = 'Angles between Fine-tuned Models'
        ratios_title = 'Model Stock Interpolation Ratios'

    # 각도 시각화
    x = list(range(len(angles)))
    ax1.scatter(x, angles, c=colors, alpha=0.7)
    ax1.set_xlabel('Parameter Index')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title(angles_title)
    ax1.grid(True, alpha=0.3)

    # 범례 추가
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Attention'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='MLP'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Other')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # 비율 시각화
    ax2.scatter(x, ratios, c=colors, alpha=0.7)
    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Interpolation Ratio')
    ax2.set_title(ratios_title)
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    # 저장
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'model_stock_visualization.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_file}")
    plt.close(fig)  # 메모리 관리를 위해 close
    return fig

def plot_angle_distribution(angles_dict, save_path, model1_path=None, model2_path=None, suffix=""):
    """각도 분포를 히스토그램으로 시각화합니다."""
    angles = [float(angle) for angle in angles_dict.values()]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 제목 설정
    title_suffix = suffix if suffix else ""
    if model1_path and model2_path:
        model1_name = extract_folder_name(model1_path)
        model2_name = extract_folder_name(model2_path)
        title = f'Distribution of Angles between {model1_name} and {model2_name} {title_suffix}'
    else:
        title = 'Distribution of Angles between Models'

    ax.hist(angles, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
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
    plt.close(fig)
    return fig

def create_summary_report(angles_dict, ratios_dict, save_path, model1_path=None, model2_path=None):
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
        if any(proj in key for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
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

    # 텍스트 보고서도 생성 (제목에 모델 이름 추가)
    text_report_file = os.path.join(save_path, 'model_stock_report.txt')
    with open(text_report_file, 'w') as f:
        if model1_path and model2_path:
            model1_name = extract_folder_name(model1_path)
            model2_name = extract_folder_name(model2_path)
            f.write(f"Model Stock Analysis Report: {model1_name} vs {model2_name}\n")
        else:
            f.write("Model Stock Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Parameters Analyzed: {len(angles_dict)}\n\n")
        f.write("Angle Statistics:\n")
        f.write(f" Mean: {angle_stats['mean']:.2f}°\n")
        f.write(f" Std: {angle_stats['std']:.2f}°\n")
        f.write(f" Min: {angle_stats['min']:.2f}°\n")
        f.write(f" Max: {angle_stats['max']:.2f}°\n")
        f.write(f" Median: {angle_stats['median']:.2f}°\n\n")
        f.write("Ratio Statistics:\n")
        f.write(f" Mean: {ratio_stats['mean']:.4f}\n")
        f.write(f" Std: {ratio_stats['std']:.4f}\n")
        f.write(f" Min: {ratio_stats['min']:.4f}\n")
        f.write(f" Max: {ratio_stats['max']:.4f}\n")
        f.write(f" Median: {ratio_stats['median']:.4f}\n\n")
        f.write("Layer-wise Analysis:\n")
        for layer_type, stats in report['layer_analysis'].items():
            f.write(f" {layer_type.upper()}:\n")
            f.write(f" Count: {stats['count']}\n")
            f.write(f" Angle Mean: {stats['angle_mean']:.2f}°\n")
            f.write(f" Ratio Mean: {stats['ratio_mean']:.4f}\n\n")

    print(f"Report saved to {report_file} and {text_report_file}")
    return report

def load_full_weights(model_path, base_weights, is_lora=True):
    """모델 경로에서 full weights를 로드하거나 재구성합니다."""
    if is_lora:
        # LoRA adapter인 경우
        print(f"Loading LoRA and reconstructing full weights from {extract_folder_name(model_path)}")
        lora_weights = load_lora_weights(model_path)
        lora_mapping = create_lora_mapping(base_weights, lora_weights)
        lora_deltas = compute_lora_delta(lora_weights, lora_mapping)
        full_weights = reconstruct_full_weights(base_weights, lora_deltas)
    else:
        # Full model인 경우
        print(f"Loading full model weights from {extract_folder_name(model_path)}")
        full_weights = {}
        # safetensors 우선 시도
        safetensors_path = os.path.join(model_path, 'pytorch_model.safetensors')  # 또는 'model.safetensors'
        if os.path.exists(safetensors_path):
            full_weights = load_file(safetensors_path, device="cpu")
            full_weights = {k: v.float() if v.dtype == torch.float16 else v for k, v in full_weights.items()}
        else:
            # pytorch bin 시도
            bin_path = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(bin_path):
                full_weights = torch.load(bin_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"No valid model files found in {model_path}")
        print(f"Loaded {len(full_weights)} parameters")
    return full_weights

def main(finetuned_path_1, finetuned_path_2, merged_path=None, average_path=None, save_path=None, k=2):
    """
    LoRA 모델들 간의 각도와 비율을 계산하고 시각화하는 범용 함수
    """
    
    # 기본 저장 경로 설정
    if save_path is None:
        save_path = './visualizations/'
    
    # 저장 경로가 '/'로 끝나지 않으면 추가
    if not save_path.endswith('/'):
        save_path += '/'
    
    try:
        # Base weights 로드 (모든 계산에 필요)
        base_model_path = '../stock_test_output/base/'  # 기본 경로; 필요 시 파라미터화
        print("Loading base model weights...")
        base_weights = load_base_model_weights(base_model_path)
        
        # 필수 모델 full weights 로드/재구성 (finetuned는 LoRA로 가정)
        print("Loading model weights...")
        full1 = load_full_weights(finetuned_path_1, base_weights, is_lora=True)
        full2 = load_full_weights(finetuned_path_2, base_weights, is_lora=True)
        
        # 선택적 모델
        full_merge = None
        full_average = None
        
        if merged_path:
            full_merge = load_full_weights(merged_path, base_weights, is_lora=False)  # merged는 full로 가정
            
        if average_path:
            full_average = load_full_weights(average_path, base_weights, is_lora=False)  # average도 full로 가정

        # 기본 각도 계산 (두 모델 간)
        print("Computing angles between the two models...")
        angles_dict = compute_angles_model_stock(full1, full2, base_weights)
        
        if not angles_dict:
            print("No angles computed. Check your model files.")
            return
        
        # 기본 비율 계산
        print("Computing ratios...")
        ratios_dict = compute_interpolation_ratios(angles_dict, k=k)
        
        # 기본 시각화 (폴더명 포함)
        print("Creating basic visualizations...")
        visualize_angles_and_ratios(angles_dict, ratios_dict, save_path, 
                                   finetuned_path_1, finetuned_path_2)
        plot_angle_distribution(angles_dict, save_path, 
                               finetuned_path_1, finetuned_path_2)
        
        # 추가 분석 (병합 모델이 있는 경우)
        if full_merge:
            print("Computing angles with merged model...")
            
            angles_m1 = compute_angles_model_stock(full1, full_merge, base_weights)
            angles_m2 = compute_angles_model_stock(full2, full_merge, base_weights)
            
            ratios_m1 = compute_interpolation_ratios(angles_m1, k=k)
            ratios_m2 = compute_interpolation_ratios(angles_m2, k=k)
            
            # 병합 모델 관련 시각화 (폴더명 포함)
            merge_save_path = f"{save_path}comparison_with_merge/"
            
            visualize_angles_and_ratios(angles_m1, ratios_m1, merge_save_path,
                                       finetuned_path_1, merged_path, "with merged model")
            
            plot_angle_distribution(angles_m1, merge_save_path,
                                   finetuned_path_1, merged_path, "with merged model")
        
        # 추가 분석 (평균 모델이 있는 경우)
        if full_average:
            print("Computing angles with average model...")
            
            angles_a1 = compute_angles_model_stock(full1, full_average, base_weights)
            angles_a2 = compute_angles_model_stock(full2, full_average, base_weights)
            
            ratios_a1 = compute_interpolation_ratios(angles_a1, k=k)
            ratios_a2 = compute_interpolation_ratios(angles_a2, k=k)
            
            # 평균 모델 관련 시각화 (폴더명 포함)
            avg_save_path = f"{save_path}comparison_with_average/"
            
            visualize_angles_and_ratios(angles_a1, ratios_a1, avg_save_path,
                                       finetuned_path_1, average_path, "with average model")
            
            plot_angle_distribution(angles_a1, avg_save_path,
                                   finetuned_path_1, average_path, "with average model")
        
        # 병합 모델과 평균 모델 간 분석 (둘 다 있는 경우)
        if full_merge and full_average:
            print("Computing angles between merged and average models...")
            
            angles_ma = compute_angles_model_stock(full_average, full_merge, base_weights)
            ratios_ma = compute_interpolation_ratios(angles_ma, k=k)
            
            ma_save_path = f"{save_path}merged_vs_average/"
            
            visualize_angles_and_ratios(angles_ma, ratios_ma, ma_save_path,
                                       average_path, merged_path)
            plot_angle_distribution(angles_ma, ma_save_path,
                                   average_path, merged_path)
        
        # 보고서 생성 (폴더명 포함)
        print("Generating summary report...")
        create_summary_report(angles_dict, ratios_dict, save_path, 
                             finetuned_path_1, finetuned_path_2)
        
        print("🎉 All visualizations and reports completed!")
        print(f"📁 Results saved to: {save_path}")
        
        return {
            'angles_basic': angles_dict,
            'ratios_basic': ratios_dict,
            'status': 'completed'
        }
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        raise

if __name__ == "__main__":
    # 예시 호출 (사용자 쿼리와 유사)
    results = main(
        finetuned_path_1='/Users/juheon/Desktop/stock_test_output/first_model/',
        finetuned_path_2='/Users/juheon/Desktop/stock_test_output/old_second_model/',
        merged_path='/Users/juheon/Desktop/stock_test_output/stock_with_full/',
        save_path='/Users/juheon/Desktop/stock_test_output/visualizations/'
    )
