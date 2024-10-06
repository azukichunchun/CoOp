import os
import re
import sys
import pdb
import numpy as np
from collections import defaultdict
from scipy.stats import hmean
import matplotlib.pyplot as plt

args = sys.argv

# ディレクトリ構造の基本パス
shots = ["shots_1"]
seeds = ["seed1", "seed2","seed3"]

#trainers = ["CoOp", "CoCoOp", "DoCoOp", "DoCoOp2", "DoCoCoOp"]
trainers = ["OneShot_Adapter_Diverse"]

datasets = ["eurosat", "dtd", "caltech101",
            "oxford_pets", "oxford_flowers", "fgvc_aircraft",
            "food101", "ucf101", "sun397", "stanford_cars", "imagenet"]

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
    differences = {}
    for dataset in datasets:
        print(f"###{dataset}###")
        base_paths = [
            f"output/base2new/train_base/{dataset}/",
            f"output/base2new/test_new/{dataset}/"
            ]
        
        fullres = dict()    
        dump = []

        for trainer in trainers:
            res = defaultdict(list)
            print(f"##{trainer}##")
            
            if trainer == "CoCoOp":
                cfg = "vit_b16_c4_ep10_batch1_ctxv1_zhou_2"
                # cfg = "vit_b16_c4_ep55_batch1_ctxv1_zhou"
            elif trainer == "CoOp":
                cfg = "vit_b16_ctxv1_use_full_class_zhou"
            elif trainer == "DoCoOp":
                cfg = "vit_b16_ep200_ctxv1_zhou"
                trainer = "DoCoOp"
            elif trainer == "DoCoCoOp":
                # cfg = "vit_b16_c4_ep10_batch1_ctxv1_zhou"
                # cfg = "vit_b16_c4_ep10_batch1_ctxv1_zhou_2"
                cfg = "vit_b16_c4_ep10_batch1_ctxv1_zhou_active"
            elif trainer == "OneShot_Adapter":
                cfg = "vit_b16"
            elif trainer == "CLIP_Adapter":
                cfg = "vit_b16"
            elif trainer == "OneShot_Adapter_Diverse":
                cfg = "vit_b16"

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
                            else:
                                print(f"Could not find accuracy in {log_path}")
                        #else:
                        #    print(f"{log_path} does not exist!")
                    
                    acc_mean = np.round(np.mean(accuracies),2)
                    acc_std = np.round(np.std(accuracies), 2)
                    
                    print(f"{base_path.split('/')[2]},{dataset},{trainer},{cfg},{shot}: -> ${acc_mean}_{{{acc_std}}}$")    
                    res[base_path.split('/')[2]+"_means"].append(acc_mean)
                    res[base_path.split('/')[2]+"_errors"].append(acc_std)
            
            harmonic_means_elementwise = [hmean([res['test_new_means'][i], res['train_base_means'][i]]) for i in range(len(res['test_new_means']))]
            for i, v in enumerate(harmonic_means_elementwise):
                print(f"{trainer}, {shots[i]}: {round(v, 2)}")
            
            res["H"] = harmonic_means_elementwise

            fullres[trainer] = res
        
        # # 16shotと1shotの精度の差を取得
        # for trainer, res in fullres.items():
        #     if trainer == "CoOp" and 'test_new_means' in res:
        #         # 16shotのindex = 4, 1shotのindex = 0
        #         diff = res['test_new_means'][4] - res['test_new_means'][0]
        #         differences[(dataset, trainer)] = diff
        #         print(f"Difference in accuracy for {dataset} and {trainer} = {diff}%")
        
        summary[dataset] = fullres
    
        # # Plotting 
        # plt.figure(figsize=(3, 5))   
        # for k, res in fullres.items():
        
        #     s = [1, 2, 4, 8, 16]
        #     base_means = res['train_base_means']
        #     base_errors = res['train_base_errors']
        #     new_means = res['test_new_means']
        #     new_errors = res['test_new_errors']
        #     H = res['H']
        #     #plt.plot(s, new_means, label=f'{k}', marker='*')
        #     plt.plot(s, H, label=f'{k}', marker='*')

        # # Labelling
        # plt.xlabel('shots')
        # plt.ylabel('accuracy')
        # plt.title(f'{dataset}')
        # #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
        # plt.legend(loc='upper left', borderaxespad=0, fontsize=10)
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # plt.tight_layout()

        # filename = dataset + ".png"

        # plt.savefig(os.path.join('output', 'plots', filename))
        
           
    # # 新しいグラフを作成
    # labels = list(differences.keys())
    # labels = [x[0] for x in labels]
    # values = list(differences.values())
    # plt.figure(figsize=(10, 6))

    # plt.bar(labels, values, label='CoOp')

    # plt.xlabel('Dataset', fontweight='bold')
    # plt.ylabel('Difference in accuracy', fontweight='bold')
    # plt.title('Difference in Accuracy Between 16shot and 1shot for CoOp')
    # plt.xticks(rotation=90)
    # plt.legend()
    # plt.tight_layout()

    # plt.savefig(os.path.join('output', 'plots', 'accuracy_difference.png'))

    # Across all dataset
    # base = []
    # new = []
    # hm = []
    # for k, v in summary.items():
    #     base += [v["CoOp"]["train_base_means"][0]]
    #     new += [v["CoOp"]["test_new_means"][0]]
    #     hm += [v["CoOp"]["H"][0]]
    # #pdb.set_trace()
    # print(f"base: {round(np.nanmean(base),2)}, \
    #         new: {round(np.nanmean(new),2)}, \
    #         H:   {round(np.nanmean(hm), 2)}")
    
    # # Rearrange and plot
    # plt.figure(figsize=(3, 5))
    # s = [1, 2, 4, 8, 16]
        
    # total = defaultdict(list)
    # for trainer in trainers:
    #     for idx, (k, v) in enumerate(summary.items()):
    #         accuracy_by_dataset = v[trainer]["train_base_means"]
            
    #         if np.isnan(accuracy_by_dataset).any():
    #             continue
            
    #         if idx == 0:
    #             total[trainer] = accuracy_by_dataset
    #         else:
    #             total[trainer] = [(x + y) / 2 for x, y in zip(total[trainer], accuracy_by_dataset)]
                
    #     plt.plot(s, total[trainer], label=f'{trainer}', marker='*')
       
    # # Labelling
    # plt.xlabel('shots')
    # plt.ylabel('accuracy')
    # plt.xlim(0, 3)
    # plt.ylim(63, 70)
    # plt.title(f'{dataset}')
    # plt.legend(loc='upper left', borderaxespad=0, fontsize=10)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.tight_layout()

    # filename = "average.png"

    # plt.savefig(os.path.join('output', 'plots', filename))            
    

# スクリプトを実行
main()