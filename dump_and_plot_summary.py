import os
import re
import sys
import pdb
import numpy as np
from collections import defaultdict
from scipy.stats import hmean
import matplotlib.pyplot as plt

args = sys.argv

#dataset = args[1]

# ディレクトリ構造の基本パス
shots = ["shots_1","shots_2","shots_4","shots_8","shots_16"]
seeds = ["seed1"]#, "seed2","seed3","seed4","seed5"]

trainers = ["CoOp", "CoCoOp", "DoCoOp", "DoCoCoOp"]
#trainers = ["CoOp", "CoCoOp", "DoCoOp"]

datasets = ["eurosat", "dtd", "isic", "caltech101",
            "oxford_pets", "oxford_flowers", "fgvc_aircraft",
            "food101", "ucf101", "sun397"]

cfgs = {"CoOp":["vit_b16_ctxv1_use_full_class",
                "vit_b16_ep50_ctxv1"],
        "CoCoOp":["vit_b16_c4_ep10_batch1_ctxv1_use_full_class",
                  "vit_b16_c4_ep10_batch1_ctxv1"],
        "DoCoOp":["vit_b16_ctxv1_ep200_reduce_use_full_class_weight_adjust",
                  "vit_b16_ctxv1_ep50_reduce_maxloss_weight_adjust"],
        "DoCoCoOp":["vit_b16_c4_ep10_batch1_ctxv1_use_full_class_adjust_weight",
                    "vit_b16_c4_ep10_batch1_ctxv1_adjust_weight"]
        }

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
    summary = dict()         
    for dataset in datasets:
        print(f"###{dataset}###")
        fullres = dict()
        base_paths = [
            f"output/base2new/test_new/{dataset}/",
            f"output/base2new/train_base/{dataset}/"
            ]
        for trainer in trainers:
            print(f"###{trainer}###")

            cfg_list = cfgs[trainer]
            res_cfg = defaultdict(list)
            
            for cfg in cfg_list:
                print(f"###{cfg}###")
                res = defaultdict(list)
                for base_path in base_paths:
                    for shot in shots:
                        accuracies = []
                        for seed in seeds:
                            log_path = os.path.join(base_path, shot, trainer, cfg, seed, "log.txt")
                            if os.path.exists(log_path):
                                accuracy = extract_accuracy_from_log(log_path)
                                if accuracy is not None:
                                    #print(f"{log_path} -> accuracy: {accuracy}%")
                                    accuracies.append(accuracy)
                        
                                if accuracies:
                                    acc_mean = np.round(np.mean(accuracies),2)
                                    acc_std = np.round(np.std(accuracies), 2)
                                else:
                                    acc_mean = acc_std = 0
                                    
                        print(f"{base_path.split('/')[2]},{dataset},{trainer},{cfg},{shot}: -> ${acc_mean}_{{{acc_std}}}$")    
                        
                        res[base_path.split('/')[2]+"_means"].append(acc_mean)
                res_cfg[cfg] = res
                #pdb.set_trace()
                    
            fullres[trainer] = res_cfg
        summary[dataset] = fullres
    #pdb.set_trace()
    # graph
    for dataset, v1 in summary.items():
        plt.figure(figsize=(6, 5))   
        for trainer, v2 in v1.items():
            dump = []
            for cfg, v3 in v2.items():
                dump.append(v3["test_new_means"])
            s = [1, 2, 4, 8, 16]
            averages  = [(a+b)/2 for a,b in zip(dump[0], dump[1])]
            plt.plot(s, averages, label=f'{trainer}(ave)', marker='*')
       
        filename = dataset + "_ave" + ".png"
        plt.savefig(os.path.join('output', 'plots', filename))
# スクリプトを実行
main()