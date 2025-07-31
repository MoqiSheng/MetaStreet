import itertools
import os
import random
import math
import torch
import numpy as np
import argparse
import warnings
from datetime import datetime
from model import SV_GAT  

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser.add_argument('--device', type=str, default='cuda:2', help='gpu device ids')
parser.add_argument('--print_num', type=int, default=1, help='gap of print evaluations')
parser.add_argument("--print_epoch", type=int, default=0, help="Start print epoch")
parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch")
parser.add_argument("--current_epoch", type=int, default=0, help="Current epoch")
parser.add_argument("--epochs", type=int, default=200, help="Epochs")
parser.add_argument("--seed", type=int, default=42, help="random seed.")
parser.add_argument("--rounds", type=int, default=5, help="number of training rounds")
parser.add_argument("--mode", type=str, default='lstm', help="aggression function.")
# parser.add_argument("--pretrain_scn_path", type=str, default='', help="Path to pretrained scene embeddings")

args = parser.parse_args()

def trainer(args, model, optimizer1, optimizer2, optimizer3, optimizer4, epoch):
    loss_epoch = []
    loss_su_epoch = []
    infonce_epoch = []
    
    model.train()
    optimizer1.zero_grad() # attention
    optimizer2.zero_grad() # sv_agg(lstm)
    optimizer3.zero_grad() # gat
    optimizer4.zero_grad() # mlp

    total_loss, loss_su, infonce_loss, pre_out, street_embedding = model()
    
    loss_epoch.append(total_loss.item())
    loss_su_epoch.append(loss_su.item())
    infonce_epoch.append(infonce_loss.item())

    total_loss.backward()

    optimizer1.step()
    optimizer2.step()
    optimizer3.step()
    optimizer4.step()

    if epoch % args.print_num == 0:
        print(f"TrainEpoch [{epoch + 1}/{args.epochs}]\t total_loss:{np.mean(loss_epoch):.6f}\t loss_su:{np.mean(loss_su_epoch):.6f}\t infonce_loss:{np.mean(infonce_epoch):.6f}")
    
    return np.mean(loss_epoch), pre_out, street_embedding

def test(args, model, epoch, round_num, result_dir):
    with torch.no_grad():
        model.eval()
        _, _, _, out, _ = model()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(result_dir, exist_ok=True)
        if args.downstream == 'poi':
            acc, f1, mrr, _, _, _, num, pred_out = model.test(out)
            result = {
                'epoch': epoch + 1,
                'acc': acc,
                'f1': f1,
                'mrr': mrr,
                'num': num
            }
            return acc, f1, mrr, 1, 1, 1, num, pred_out, result
        else:
            a1, a3, a5, a10, f1, mrr, num, pred_out = model.test(out)
            result = {
                'epoch': epoch + 1,
                'a1': a1,
                'a3': a3,
                'a5': a5,
                'a10': a10,
                'f1': f1,
                'mrr': mrr,
                'num': num
            }
            return a1, a3, a5, a10, f1, mrr, num, pred_out, result

def calculate_best_results(result_dir, downstream):
    round_files = [f for f in os.listdir(result_dir) if f.startswith('round_') and f.endswith('.npy')]
    if not round_files:
        return None

    epoch_results = {}
    for round_file in round_files:
        round_data = np.load(os.path.join(result_dir, round_file), allow_pickle=True).item()
        for epoch_data in round_data['epochs']:
            epoch = epoch_data['epoch']
            if epoch not in epoch_results:
                epoch_results[epoch] = []
            epoch_results[epoch].append(epoch_data)

    epoch_averages = {}
    best_f1 = -1
    best_epoch = None

    if downstream == 'poi':
        all_results = []
        for epoch in epoch_results:
            accs = [r['acc'] for r in epoch_results[epoch]]
            f1s = [r['f1'] for r in epoch_results[epoch]]
            mrrs = [r['mrr'] for r in epoch_results[epoch]]
            epoch_averages[epoch] = {
                'avg_acc': np.mean(accs),
                'avg_f1': np.mean(f1s),
                'avg_mrr': np.mean(mrrs)
            }
            all_results.extend(epoch_results[epoch])
            if epoch_averages[epoch]['avg_f1'] > best_f1:
                best_f1 = epoch_averages[epoch]['avg_f1']
                best_epoch = epoch

        best_results = {
            'best_epoch': best_epoch,
            'best_epoch_avg_acc': epoch_averages[best_epoch]['avg_acc'],
            'best_epoch_avg_f1': epoch_averages[best_epoch]['avg_f1'],
            'best_epoch_avg_mrr': epoch_averages[best_epoch]['avg_mrr'],
            'overall_best_acc': max([r['acc'] for r in all_results]),
            'overall_best_f1': max([r['f1'] for r in all_results]),
            'overall_best_mrr': max([r['mrr'] for r in all_results])
        }
    else:
        all_results = []
        for epoch in epoch_results:
            a1s = [r['a1'] for r in epoch_results[epoch]]
            a3s = [r['a3'] for r in epoch_results[epoch]]
            a5s = [r['a5'] for r in epoch_results[epoch]]
            a10s = [r['a10'] for r in epoch_results[epoch]]
            f1s = [r['f1'] for r in epoch_results[epoch]]
            mrrs = [r['mrr'] for r in epoch_results[epoch]]
            epoch_averages[epoch] = {
                'avg_a1': np.mean(a1s),
                'avg_a3': np.mean(a3s),
                'avg_a5': np.mean(a5s),
                'avg_a10': np.mean(a10s),
                'avg_f1': np.mean(f1s),
                'avg_mrr': np.mean(mrrs)
            }
            all_results.extend(epoch_results[epoch])
            if epoch_averages[epoch]['avg_f1'] > best_f1:
                best_f1 = epoch_averages[epoch]['avg_f1']
                best_epoch = epoch

        best_results = {
            'best_epoch': best_epoch,
            'best_epoch_avg_a1': epoch_averages[best_epoch]['avg_a1'],
            'best_epoch_avg_a3': epoch_averages[best_epoch]['avg_a3'],
            'best_epoch_avg_a5': epoch_averages[best_epoch]['avg_a5'],
            'best_epoch_avg_a10': epoch_averages[best_epoch]['avg_a10'],
            'best_epoch_avg_f1': epoch_averages[best_epoch]['avg_f1'],
            'best_epoch_avg_mrr': epoch_averages[best_epoch]['avg_mrr'],
            'overall_best_a1': max([r['a1'] for r in all_results]),
            'overall_best_a3': max([r['a3'] for r in all_results]),
            'overall_best_a5': max([r['a5'] for r in all_results]),
            'overall_best_a10': max([r['a10'] for r in all_results]),
            'overall_best_f1': max([r['f1'] for r in all_results]),
            'overall_best_mrr': max([r['mrr'] for r in all_results])
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f'{result_dir}/epoch_averages_{timestamp}.npy', epoch_averages)
    np.save(f'{result_dir}/best_results_{timestamp}.npy', best_results)
    return best_results

def run_training(pretrain_sv_path, result_subdir):
    result_dir = f'result/{result_subdir}'
    os.makedirs(result_dir, exist_ok=True)

    for downstream in ['function', 'poi']:
        print(f"\nStarting {downstream} downstream with {pretrain_sv_path}")
        args.downstream = downstream
        args.pretrain_sv_path = pretrain_sv_path
        if downstream == 'function':
            args.pretrain_scn_path = 'embeddings/qwen_text_embedding_function_72.pt'
        else:  # downstream == 'poi'
            args.pretrain_scn_path = 'embeddings/qwen_text_embedding_poi_72.pt'
        args.current_epoch = 0

        for round_num in range(args.rounds):
            print(f"Round {round_num + 1}/{args.rounds}")

            np.random.seed(args.seed + round_num)
            random.seed(args.seed + round_num + 1)
            torch.manual_seed(args.seed + round_num + 2)
            torch.cuda.manual_seed(args.seed + round_num + 3)
            torch.backends.cudnn.deterministic = True

            model = SV_GAT(args)
            model = model.to(args.device)

            opt1 = torch.optim.Adam(
                itertools.chain(model.attention_soft.parameters()),
                lr=0.0005, weight_decay=1e-8)
            opt4 = torch.optim.Adam(
                model.text_mlp.parameters(),
                lr=0.0005, weight_decay=1e-8) 
            if args.downstream == 'poi':
                opt3 = torch.optim.Adam(model.gat_poi.parameters(), lr=0.0005, weight_decay=5e-4)
                args.epochs = 150
            else:
                opt3 = torch.optim.Adam(model.gat.parameters(), lr=0.005, weight_decay=5e-4)
                args.epochs = 200

            if args.mode != 'mean':
                opt2 = torch.optim.SGD(model.sv_agg.parameters(), lr=0.005, weight_decay=1e-4, momentum=0.9)
                t = 10
                T = 800
                n_t = 0.5
                lf = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
                        1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
                        1 + math.cos(math.pi * (epoch - t) / (T - t)))
                scheduler = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda=lf)
            else:
                opt2 = torch.optim.SGD(model.sv_agg.parameters(), lr=0.005, weight_decay=1e-4, momentum=0.9)

            print(model)

            round_results = {'round': round_num, 'epochs': []}

            for epoch in range(args.start_epoch, args.epochs):
                loss_epoch, pred_, street_embedding = trainer(args, model, opt1, opt2, opt3, opt4, epoch)
                if args.mode != 'mean':
                    scheduler.step()
                if epoch % args.print_num == 0:
                    result_tuple = test(args, model, epoch, round_num, f'{result_dir}/{downstream}')
                    round_results['epochs'].append(result_tuple[-1])  

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            np.save(f'{result_dir}/{downstream}/round_{round_num}_{timestamp}.npy', round_results)

        best_results = calculate_best_results(f'{result_dir}/{downstream}', downstream)
        print(f"Best Results for {downstream}:", best_results)

run_training('embeddings/image_representation_117144_16.pt', 'MetaStreet')