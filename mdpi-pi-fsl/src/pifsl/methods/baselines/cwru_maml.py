from pifsl.data.cwru.utils import setup_logger

import numpy as np
import random
import matplotlib.pyplot as plt

import argparse
import os
import torch
import learn2learn as l2l
import torch.nn.functional as F

import csv
from datetime import datetime
from pathlib import Path

RESULTS_CSV = Path("artifacts/results/cwru_maml_results.csv")
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

METRIC_FIELDS = [
    "timestamp",
    "experiment",
    "dataset",
    "method",
    "source",
    "target",
    "extra",
    "train_acc",
    "test_acc",
    "train_err",
    "test_err",
]

def append_metrics(row: dict) -> None:
    file_exists = RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if not file_exists:
            writer.writeheader()
        full_row = {k: row.get(k, "") for k in METRIC_FIELDS}
        writer.writerow(full_row)

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways)*2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

def plot_metrics(args, iteration, train_acc, test_acc, train_loss, test_loss, experiment_title):
    plt.figure()
    plt.plot(train_acc, '-o', label="train acc")
    plt.plot(test_acc, '-o', label="test acc")
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title("Accuracy Curve by Iteration")
    plt.legend()
    plt.savefig(args.plot_path + '/' + experiment_title + '_{}.png'.format(iteration))
    plt.show()

    plt.figure()
    plt.plot(train_loss, '-o', label="train loss")
    plt.plot(test_loss, '-o', label="test loss")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title("Loss Curve by Iteration")
    plt.legend()
    plt.savefig(args.plot_path + '/' + experiment_title + '_{}.png'.format(iteration))
    plt.show()

def print_logs(iteration, train_loss, train_acc, test_loss, test_acc):
    print('Iteration {}: train_loss {:.4f}, train_acc {:.4f}, test_loss {:.4f}, test_acc {:.4f}'
          .format(iteration, train_loss, train_acc, test_loss, test_acc))

def train_model(args, model, maml, opt, loss, 
                train_tasks, test_tasks, 
                device, 
                experiment_title):
    train_acc_list = []
    train_err_list = []
    test_acc_list = []
    test_err_list = []

    for iteration in range(args.iters):
        opt.zero_grad()
        meta_train_err_sum = 0.0
        meta_train_acc_sum = 0.0
        meta_test_err_sum = 0.0
        meta_test_acc_sum = 0.0

        train_index = random.randint(0, len(args.train_domains)-1)

        for task in range(args.meta_batch_size):
            learner = maml.clone()
            batch = train_tasks[train_index].sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            evaluation_error.backward()
            meta_train_err_sum += evaluation_error.item()
            meta_train_acc_sum += evaluation_accuracy.item()

            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            evaluation_error.backward()
            meta_test_err_sum += evaluation_error.item()
            meta_test_acc_sum += evaluation_accuracy.item()

        meta_train_acc = meta_train_acc_sum / args.meta_batch_size
        meta_train_err = meta_train_err_sum / args.meta_batch_size
        meta_test_err = meta_test_err_sum / args.meta_batch_size
        meta_test_acc = meta_test_acc_sum / args.meta_batch_size

        train_acc_list.append(meta_train_acc)
        test_acc_list.append(meta_test_acc)
        train_err_list.append(meta_train_err)
        test_err_list.append(meta_test_err)

        if args.plot and iteration % args.plot_step == 0:
            plot_metrics(args, 
                         iteration, 
                         train_acc_list, test_acc_list, 
                         train_err_list, test_err_list, 
                         experiment_title)

        if args.checkpoint and iteration % args.checkpoint_step == 0:
            torch.save(model.state_dict(), 
                       args.checkpoint_path + '/' +
                       experiment_title + 
                       '_{}.pt'.format(iteration))
        if args.log:
            print_logs(iteration, meta_train_err, meta_train_acc, meta_test_err, meta_test_acc)

        # Mean gradients across meta-batch tasks to keep step size batch-invariant.
        for p in model.parameters():
            p.grad.data.mul_(1.0 / args.meta_batch_size)
        opt.step()

    append_metrics(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment": experiment_title,
            "dataset": args.dataset.lower(),          # canonical dataset id
            "method": "maml",
            "source": "domains=" + ",".join(str(d) for d in args.train_domains),
            "target": f"domain={args.test_domain}",
            "extra": f"ways={args.ways},shots={args.shots},preprocess={args.preprocess}",
            "train_acc": float(train_acc_list[-1]) if train_acc_list else "",
            "test_acc": float(test_acc_list[-1]) if test_acc_list else "",
            "train_err": float(train_err_list[-1]) if train_err_list else "",
            "test_err": float(test_err_list[-1]) if test_err_list else "",
        }
    )

def train(args, experiment_title):
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = l2l.vision.models.ConvBase(output_size=args.ways,
                                      hidden=64,
                                      channels=1,
                                      max_pool=True,
                                      embedding_size=64)
    model.to(device)

    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=args.first_order)
    opt = torch.optim.Adam(maml.parameters(), lr=args.meta_lr)

    loss = F.cross_entropy

    train_tasks = []
    for domain in args.train_domains:
        taskset = l2l.data.TaskDataset(
            l2l.vision.datasets.CWRU(args.data_dir_path,
                                     domain=domain,
                                     preprocess=args.preprocess),
            task_transforms=[
                l2l.data.transforms.NWays(taskset=None, n=args.ways),
                l2l.data.transforms.KShots(taskset=None, k=2*args.shots),
                l2l.data.transforms.LoadData(taskset=None),
                l2l.data.transforms.RemapLabels(taskset=None),
                l2l.data.transforms.ConsecutiveLabels(taskset=None),
            ],
            num_tasks=args.train_task_num,
        )
        train_tasks.append(taskset)

    test_tasks = l2l.data.TaskDataset(
        l2l.vision.datasets.CWRU(args.data_dir_path,
                                 domain=args.test_domain,
                                 preprocess=args.preprocess),
        task_transforms=[
            l2l.data.transforms.NWays(taskset=None, n=args.ways),
            l2l.data.transforms.KShots(taskset=None, k=2*args.shots),
            l2l.data.transforms.LoadData(taskset=None),
            l2l.data.transforms.RemapLabels(taskset=None),
            l2l.data.transforms.ConsecutiveLabels(taskset=None),
        ],
        num_tasks=args.test_task_num,
    )

    if args.plot:
        if not os.path.exists(args.plot_path):
            os.makedirs(args.plot_path)

    if args.checkpoint:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)

    if args.log:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        setup_logger(args.log_path, experiment_title)

    train_model(args, model, maml, opt, loss, train_tasks, test_tasks, device, experiment_title)

def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of \
                                     Model-Agnostic Meta Learning on \
                                     Fault Diagnosis Datasets')
    parser.add_argument('--ways', type=int, default=10,
                        help='Number of classes per task, default=10')
    parser.add_argument('--shots', type=int, default=5,
                        help='Number of support examples per class, default=1')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Outer loop learning rate, default=0.001')
    parser.add_argument('--fast_lr', type=float, default=0.1,
                        help='Inner loop learning rate, default=0.1')
    parser.add_argument('--adapt_steps', type=int, default=5,
                        help='Number of inner loop steps for adaptation, default=5')
    parser.add_argument('--meta_batch_size', type=int, default=32,
                        help='Number of outer loop iterations, \
                              i.e. no. of meta-tasks for each batch, \
                              default=32')
    parser.add_argument('--iters', type=int, default=300,
                        help='Number of outer-loop iterations, default=300')
    parser.add_argument('--first_order', type=bool, default=True,
                        help='Use the first-order approximation, default=True')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Use CUDA if available, default=True')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed, default=42')
    parser.add_argument('--data_dir_path', type=str, default='./data',
                        help='Path to the data directory, default=./data')
    parser.add_argument('--dataset', type=str, default='CWRU',
                        help='Which dataset to use, \
                            default=CWRU, \
                            options=[CWRU, HST]')
    parser.add_argument('--preprocess', type=str, default='STFT',
                        help='Which preprocessing technique to use, \
                            default=STFT, \
                            options=[WT, STFT, FFT]')
    parser.add_argument('--train_domains', type=str, default='0,1,2',
                        help='Training domain, integer(s) separated by commas, default=0,1,2')
    parser.add_argument('--test_domain', type=int, default=3,
                        help='Test domain, single integer, default=3')
    parser.add_argument('--train_task_num', type=int, default=200,
                        help='Number of samples per domain for training, default=200')
    parser.add_argument('--test_task_num', type=int, default=100,
                        help='Number of samples per domain for testing, default=100')
    parser.add_argument('--plot', type=bool, default=True,
                        help='Plot the learning curve, default=True')
    parser.add_argument('--plot_path', type=str, default='./images',
                        help='Directory to save the learning curve, default=./images')
    parser.add_argument('--plot_step', type=int, default=50,
                        help='Step for plotting the learning curve, default=50')
    parser.add_argument('--log', type=bool, default=True,
                        help='Log the training process, default=True')
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='Directory to save the logs, default=./logs')
    parser.add_argument('--checkpoint', type=bool, default=True,
                        help='Save the model checkpoints, default=True')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints',
                        help='Directory to save the model checkpoints, default=./checkpoints')
    parser.add_argument('--checkpoint_step', type=int, default=50,
                        help='Step for saving the model checkpoints, default=50')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset not in ['CWRU', 'HST']:
        raise ValueError('Dataset must be either CWRU or HST.')
    if args.preprocess not in ['WT', 'STFT', 'FFT']:
        raise ValueError('Preprocessing technique must be either WT, STFT, or FFT.')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.train_domains = args.train_domains.split(',')
    train_domains_str = ''
    for i in range(len(args.train_domains)):
        train_domains_str += str(args.train_domains[i])
    args.train_domains = [int(i) for i in args.train_domains]

    experiment_title = 'MAML_{}_{}_{}w{}s_source{}_target{}'.format(args.dataset, 
                                                args.preprocess,
                                                args.ways,
                                                args.shots,
                                                train_domains_str,
                                                args.test_domain)

    train(args, experiment_title)
