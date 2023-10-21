import os
import re
import sys
import pdb
import numpy as np
from collections import defaultdict
from scipy.stats import hmean
import matplotlib.pyplot as plt

args = sys.argv

dataset = args[1]
trainer = args[2]

if trainer == "CoCoOp":
    cfg = "vit_b16_c4_ep10_batch1_ctxv1"
    #cfg = "vit_b16_c4_ep10_batch1"
elif trainer == "CoOp":
    cfg = "vit_b16_ctxv1"
elif trainer == "DoCoOp":
    # cfg = "vit_b16_ctxv1"
    # cfg = "vit_b16_ctxv1_reduce"
    # cfg = "vit_b16_ctxv1_reduce_minloss"
    cfg = "vit_b16_ctxv1_lambda_4"
    #cfg = "vit_b16_ctxv1_lambda_6"
    #cfg = "vit_b16_ctxv1_no_dist_coef"
    #cfg = "vit_b16_ctxv1_no_ot"
    #cfg = "vit_b16_ctxv1_no_mi"
elif trainer == "DoCoCoOp":
    cfg = "vit_b16_c4_ep10_batch1_ctxv1"

# ディレクトリ構造の基本パス
base_paths = [f"output/base2new/test_new/{dataset}/",
              f"output/base2new/train_base/{dataset}/"]
shots = ["shots_1","shots_2","shots_4","shots_8","shots_16"]
seeds = ["seed1", "seed2","seed3","seed4","seed5"]

# accuracyを抽出する関数
def extract_accuracy_from_log(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        # 正規表現を使用してaccuracyを抽出
        match = re.search(r'accuracy: (\d+\.\d+)%', content)
        if match:
            return float(match.group(1))
    return None

# main関数
def main():
    
    res = defaultdict(list)
    
    for base_path in base_paths:
        for shot in shots:
            accuracies = []
            for seed in seeds:
                log_path = os.path.join(base_path, shot, trainer, cfg, seed, "log.txt")
                if os.path.exists(log_path):
                    accuracy = extract_accuracy_from_log(log_path)
                    if accuracy is not None:
                        print(f"{log_path} -> accuracy: {accuracy}%")
                        accuracies.append(accuracy)
                    else:
                        print(f"Could not find accuracy in {log_path}")
                else:
                    print(f"{log_path} does not exist!")
            
            acc_mean = np.round(np.mean(accuracies),2)
            acc_std = np.round(np.std(accuracies), 2)
            print(f"{base_path.split('/')[2]},{dataset},{trainer},{cfg},{shot}: -> ${acc_mean}_{{{acc_std}}}$")    
            res[base_path.split('/')[2]+"_means"].append(acc_mean)
            res[base_path.split('/')[2]+"_errors"].append(acc_std)
    
    harmonic_means_elementwise = [hmean([res['test_new_means'][i], res['train_base_means'][i]]) for i in range(len(res['test_new_means']))]
    for i, v in enumerate(harmonic_means_elementwise):
        print(f"{trainer}, {shots[i]}: {round(v, 2)}")
    
    
    # # Plotting
    s = [1, 2, 4, 8, 16]
    base_means = res['train_base_means']
    base_errors = res['train_base_errors']
    new_means = res['test_new_means']
    new_errors = res['test_new_errors']
    H = harmonic_means_elementwise
    #pdb.set_trace()
    plt.figure(figsize=(4, 4))
    plt.errorbar(s, base_means, yerr=base_errors, fmt='-o', label='base', capsize=5,  capthick=1.5)
    plt.errorbar(s, new_means, yerr=new_errors, fmt='-s', label='new', capsize=5,  capthick=1.5)
    plt.scatter(s, H, color='red', marker='D', label='H', zorder=3)

    # Labelling
    plt.xlabel('shots')
    plt.ylabel('accuracy')
    plt.title('Comparison of base, new and H')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    filename = trainer + "_" + dataset + ".png"

    plt.savefig(os.path.join('output', 'plots', filename))
    
# スクリプトを実行
main()