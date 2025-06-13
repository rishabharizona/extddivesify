# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader
from shap_utils import (
    get_shap_explainer,
    compute_shap_values,
    _get_shap_array,
    plot_summary,
    plot_force,
    evaluate_shap_impact,
    plot_shap_heatmap,
    get_background_batch,
    compute_jaccard_topk,
    compute_kendall_tau,
    cosine_similarity_shap,
    log_shap_numpy,
    overlay_signal_with_shap
)
from shap_utils_extended import (
    compute_flip_rate,
    compute_confidence_change,
    compute_aopc,
    compute_feature_coherence,
    compute_shap_entropy
)

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)
    if args.latent_domain_num < 6:
        args.batch_size = 32*args.latent_domain_num
    else:
        args.batch_size = 16*args.latent_domain_num

    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)

    best_valid_acc, target_acc = 0, 0
    logs = {
        'epoch': [],
        'class_loss': [],
        'dis_loss': [],
        'total_loss': [],
        'train_acc': [],
        'valid_acc': [],
        'target_acc': [],
        'total_cost_time': []
    }

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    for round in range(args.max_epoch):
        print(f'\n========ROUND {round}========')
        print('====Feature update====')
        loss_list = ['class']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step]+[loss_result_dict[item] for item in loss_list], colwidth=15)

        print('====Latent domain characterization====')
        loss_list = ['total', 'dis', 'ent']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step]+[loss_result_dict[item] for item in loss_list], colwidth=15)

        algorithm.set_dlabel(train_loader)

        print('====Domain-invariant feature learning====')
        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch']
        print_key.extend([item+'_loss' for item in loss_list])
        print_key.extend([item+'_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15)

        sss = time.time()
        for step in range(args.local_epoch):
            for data in train_loader:
                step_vals = algorithm.update(data, opt)

            results = {
                'epoch': round * args.local_epoch + step,
                'train_acc': modelopera.accuracy(algorithm, train_loader_noshuffle, None),
                'valid_acc': modelopera.accuracy(algorithm, valid_loader, None),
                'target_acc': modelopera.accuracy(algorithm, target_loader, None),
                'total_cost_time': time.time() - sss
            }
            for key in loss_list:
                results[key+'_loss'] = step_vals[key]

            for key in logs:
                logs[key].append(results.get(key, 0))

            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']

            print_row([results[key] for key in print_key], colwidth=15)

    print(f'Target acc: {target_acc:.4f}')

    if args.enable_shap:
        print("Running SHAP explainability...")
        
# 1. Get full signal for SHAP from entire loader
        full_inputs = []
        for batch in tqdm(valid_loader, desc="Collecting full input for SHAP overlay"):
            x = batch[0].cpu()
            full_inputs.append(x)

        full_inputs = torch.cat(full_inputs, dim=0)  # shape: (N, C, T)
        flat_signal = full_inputs.reshape(-1)

# 2. Use background from initial samples
        background = full_inputs[:64].to('cuda')
        shap_explainer = get_shap_explainer(algorithm, background)

# 3. Compute SHAP for entire full_inputs
        shap_vals = compute_shap_values(shap_explainer, full_inputs.to('cuda'))
        shap_array = _get_shap_array(shap_vals).reshape(-1)

# 4. Save long overlay
        overlay_signal_with_shap(flat_signal.numpy(), shap_array, output_path="shap_overlay_full.png")


        base_preds, masked_preds, acc_drop = evaluate_shap_impact(algorithm, X_eval, shap_vals, top_k=10)
        print(f"[SHAP] Perturbation-based accuracy drop: {acc_drop:.4f}")

        log_shap_numpy(shap_vals, save_path="shap_values.npy")

        if len(shap_array) > 1:
            jaccard = compute_jaccard_topk(shap_array[0], shap_array[1], k=10)
            tau = compute_kendall_tau(shap_array[0], shap_array[1])
            cos_sim = cosine_similarity_shap(shap_array[0], shap_array[1])
            print(f"[SHAP] Jaccard similarity (top-10): {jaccard:.4f}")
            print(f"[SHAP] Kendall’s Tau: {tau:.4f}")
            print(f"[SHAP] Cosine Similarity: {cos_sim:.4f}")

        flip_rate = compute_flip_rate(base_preds, masked_preds)
        print(f"[SHAP] Flip Rate: {flip_rate:.4f}")

        conf_delta = compute_confidence_change(base_preds, masked_preds)
        print(f"[SHAP] Confidence Change: {conf_delta:.4f}")

        aopc = compute_aopc(algorithm, X_eval, shap_vals, evaluate_shap_impact)
        print(f"[SHAP] AOPC (Area over Perturbation Curve): {aopc:.4f}")

        entropy = compute_shap_entropy(shap_array)
        print(f"[SHAP] Entropy of SHAP Distribution: {entropy:.4f}")

        coherence = compute_feature_coherence(shap_array)
        print(f"[SHAP] Feature Coherence Score: {coherence:.4f}")

    # Final training curve plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(logs['epoch'], logs['class_loss'], label="Class Loss", marker='o')
    plt.plot(logs['epoch'], logs['dis_loss'], label="Dis Loss", marker='x')
    plt.plot(logs['epoch'], logs['total_loss'], label="Total Loss", linestyle='--')
    plt.title("Losses over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(logs['epoch'], logs['train_acc'], label="Train Accuracy", marker='o')
    plt.plot(logs['epoch'], logs['valid_acc'], label="Valid Accuracy", marker='x')
    plt.plot(logs['epoch'], logs['target_acc'], label="Target Accuracy", linestyle='--')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics_plot.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    args = get_args()
    main(args)
