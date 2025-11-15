import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Tuple
from tqdm import tqdm

from Central import CentralServer
from Client import EEGNet, ACRNN, DeepConvNet, ClientAdaptiveLayer, LocalClient
from preprocess import dataset_preprocess, Dataset


def federated_learning(clients: List[LocalClient], server: CentralServer,
                       train_loaders: List[DataLoader], test_loader: DataLoader,
                       num_rounds=200, alpha_schedule='linear', training_epochs=10, sigma=0.15,
                       fisher_update_frequency=10):

    print("\n" + "=" * 60)
    print("Starting Federated Learning...")
    print("=" * 60)

    best_acc = 0
    avg_acc = 0.0
    avg_f1 = 0.0

    round_pbar = tqdm(range(num_rounds), desc='FL Rounds', position=0)

    for round_idx in round_pbar:
        if alpha_schedule == 'linear':
            alpha_t = round_idx / num_rounds
        else:
            alpha_t = 0.6

        update_fisher = (round_idx > 0 and round_idx % fisher_update_frequency == 0)

        # Local training
        client_prototypes = []

        tqdm.write(f"\n[Round {round_idx + 1}/{num_rounds}] Local Training Phase")
        if update_fisher:
            tqdm.write(f"  [EWC] Updating Fisher Information this round")

        for i, client in enumerate(clients):
            global_proto_tensor = (torch.tensor(server.global_prototypes, dtype=torch.float32)
                                   .to(client.device) if server.global_prototypes is not None
                                   else None)

            client.train_local(train_loaders[i], global_proto_tensor,
                               alpha_t=alpha_t, sigma=sigma, epochs=training_epochs,
                               show_progress=True, update_fisher=update_fisher)

            prototypes = client.get_prototypes(train_loaders[i], sigma=sigma)
            client_prototypes.append(prototypes)

        # Global aggregation
        global_prototypes = server.aggregate_prototypes(client_prototypes)

        # Distribute to clients
        for client in clients:
            client.global_prototypes = torch.tensor(global_prototypes,
                                                    dtype=torch.float32).to(client.device)

        # Evaluation
        if (round_idx + 1) % 10 == 0:
            accuracies = []
            f1_scores = []

            for i, client in enumerate(clients):
                acc, f1 = client.evaluate(test_loader)
                accuracies.append(acc)
                f1_scores.append(f1)

            avg_acc = np.mean(accuracies)
            avg_f1 = np.mean(f1_scores)

            if avg_acc > best_acc:
                best_acc = avg_acc

            tqdm.write(f"\n{'=' * 60}")
            tqdm.write(f"Round {round_idx + 1}/{num_rounds} Evaluation:")
            tqdm.write(f"  Avg Accuracy: {avg_acc:.4f}, Avg F1: {avg_f1:.4f}")
            tqdm.write(f"  Best Accuracy: {best_acc:.4f}")
            tqdm.write(f"  Client Accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
            tqdm.write(f"  Client F1 Scores: {[f'{f1:.4f}' for f1 in f1_scores]}")
            tqdm.write(f"{'=' * 60}\n")

        round_pbar.set_postfix({
            'avg_acc': f'{avg_acc:.4f}',
            'best_acc': f'{best_acc:.4f}'
        })

    return clients, best_acc


if __name__ == "__main__":
    import random

    base_seed = 42
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)

    num_classes = 2
    feature_dim = 32
    num_clients = 3
    training_epochs = 5
    batch_size = 16
    dropout_rate = 0.3
    emotion = "arousal"
    num_runs = 1

    # DEAP path
    dataset_dir = "/yourDatasetPath/"

    if not os.path.exists(dataset_dir):
        print(f"Warning: Dataset directory not found: {dataset_dir}")
        print("Please update the dataset_dir variable with your actual data path")


    try:
        if torch.cuda.is_available():
            test_tensor = torch.zeros(1).cuda()
            device = torch.device('cuda:2')
            print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device('cpu')
            print("Using device: CPU")
    except Exception as e:
        print(f"CUDA error: {e}. Falling back to CPU")
        device = torch.device('cpu')

    # DEAP subjects
    train_subjects = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10',
                      's11', 's12', 's13', 's14', 's15','s16', 's17', 's18', 's19', 's20',
                      's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30']

    test_subjects = ['s31', 's32']


    model_names = ['EEGNet_1', 'ACRNN_1','DeepConvNet_1']

    all_acc_results = [[] for _ in range(num_clients)]
    all_f1_results = [[] for _ in range(num_clients)]

    print(f"\n{'=' * 60}")
    print(f"HeteFedEEG Training on DEAP Dataset (Random Split)")
    print(f"Emotion Dimension: {emotion}")
    print(f"Number of Clients: {num_clients}")
    print(f"Models: {model_names}")
    print(f"Number of Runs: {num_runs} (with random data splits)")
    print(f"{'=' * 60}\n")

    run_pbar = tqdm(range(num_runs), desc='Overall Progress', position=0)

    for run_id in run_pbar:
        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"Run {run_id + 1}/{num_runs}")
        tqdm.write(f"{'=' * 60}")

        current_seed = base_seed + run_id
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)

        shuffled_subjects = train_subjects.copy()
        random.shuffle(shuffled_subjects)

        subjects_per_client = len(shuffled_subjects) // num_clients
        client_subjects = [
            shuffled_subjects[i * subjects_per_client:(i + 1) * subjects_per_client]
            for i in range(num_clients)
        ]

        tqdm.write(f"\nRun {run_id + 1} - Subject Assignment:")
        for i, subjects in enumerate(client_subjects):
            tqdm.write(f"  Client {i} ({model_names[i]}): {subjects}")
        tqdm.write("")

        train_loaders = []
        test_loader = None

        tqdm.write("Loading dataset...")

        loading_pbar = tqdm(enumerate(client_subjects),
                            total=len(client_subjects),
                            desc='Loading clients',
                            leave=False)

        for i, subjects in loading_pbar:
            all_data = []
            all_labels = []

            for subject in subjects:
                try:
                    # Dataset
                    data, labels = deap_preprocess(subject, emotion, dataset_dir)

                    all_data.append(data)
                    all_labels.append(labels)
                except Exception as e:
                    tqdm.write(f"Error loading {subject}: {e}")

            if all_data:
                client_data = np.concatenate(all_data, axis=0)
                client_labels = np.concatenate(all_labels, axis=0)

                indices = np.arange(len(client_data))
                np.random.shuffle(indices)
                client_data = client_data[indices]
                client_labels = client_labels[indices]


                dataset = DEAPDataset(client_data, client_labels)


                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                train_loaders.append(loader)

            loading_pbar.set_postfix({'samples': len(dataset) if all_data else 0})

        tqdm.write("\nLoading test set...")
        test_data = []
        test_labels = []

        test_pbar = tqdm(test_subjects, desc='Loading test subjects', leave=False)
        for subject in test_pbar:
            try:

                data, labels = deap_preprocess(subject, emotion, dataset_dir)

                test_data.append(data)
                test_labels.append(labels)
            except Exception as e:
                tqdm.write(f"Error loading test subject {subject}: {e}")

        if test_data:
            test_data = np.concatenate(test_data, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            test_dataset = DEAPDataset(test_data, test_labels)

            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            tqdm.write(f"Test set: Total {len(test_dataset)} samples\n")

        models = [
            EEGNet(n_timepoints=384, n_channels=32, n_classes=num_classes,
                   feature_dim=feature_dim, dropout_rate=dropout_rate),
            ACRNN(n_timepoints=384, n_channels=32, n_classes=num_classes,
                  feature_dim=feature_dim),
            DeepConvNet(n_timepoints=384, n_channels=32, n_classes=num_classes,
                        feature_dim=feature_dim),
        ]

        clients = []
        for i in range(num_clients):
            client = LocalClient(
                client_id=i,
                model=models[i],
                feature_dim=feature_dim,
                num_classes=num_classes,
                device=device,
                lr=1e-3,
                rho=0.7,
                lambda_r=0.1,
                lambda_ewc=0.01,
                tau_ewc=10
            )
            clients.append(client)

        server = CentralServer(num_classes=num_classes, feature_dim=feature_dim)

        # run
        trained_clients, best_accuracy = federated_learning(
            clients=clients,
            server=server,
            train_loaders=train_loaders,
            test_loader=test_loader,
            num_rounds=50,
            alpha_schedule='linear',
            training_epochs=training_epochs,
            sigma=0.15
        )

        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"Final Evaluation on Test Set for Run {run_id + 1}:")
        tqdm.write(f"{'=' * 60}")

        for i, client in enumerate(trained_clients):
            acc, f1 = client.evaluate(test_loader)
            all_acc_results[i].append(acc)
            all_f1_results[i].append(f1)
            tqdm.write(f"Client {i} ({model_names[i]}) -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

        current_avg_acc = np.mean([np.mean(all_acc_results[i]) for i in range(num_clients)])
        run_pbar.set_postfix({'current_avg_acc': f'{current_avg_acc:.4f}'})

    print(f"\n{'=' * 60}")
    print("Final Results Over Multiple Runs (Mean ± Std)")
    print(f"Random splits across runs for robust std estimation")
    print(f"{'=' * 60}")

    client_results = []
    for i in range(num_clients):
        acc_mean = np.mean(all_acc_results[i])
        acc_std = np.std(all_acc_results[i])
        f1_mean = np.mean(all_f1_results[i])
        f1_std = np.std(all_f1_results[i])

        client_results.append({
            'Client': i,
            'Model': model_names[i],
            'Accuracy_Mean': acc_mean,
            'Accuracy_Std': acc_std,
            'F1_Mean': f1_mean,
            'F1_Std': f1_std
        })

        print(f"Client {i} ({model_names[i]}):")
        print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"  F1-Score: {f1_mean:.4f} ± {f1_std:.4f}")

    all_accs = [acc for client_acc in all_acc_results for acc in client_acc]
    all_f1s = [f1 for client_f1 in all_f1_results for f1 in client_f1]

    overall_acc_mean = np.mean(all_accs)
    overall_acc_std = np.std(all_accs)
    overall_f1_mean = np.mean(all_f1s)
    overall_f1_std = np.std(all_f1s)

    print(f"\nOverall Performance:")
    print(f"  Accuracy: {overall_acc_mean:.4f} ± {overall_acc_std:.4f}")
    print(f"  F1-Score: {overall_f1_mean:.4f} ± {overall_f1_std:.4f}")

    result_dir = "./hetefed_results/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"\nCreated result directory: {result_dir}")

    results_df = pd.DataFrame(client_results)
    results_file = os.path.join(result_dir, f'hetefed_deap_{emotion}_results_6clients_random_split.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")

    overall_df = pd.DataFrame([{
        'Overall_Accuracy_Mean': overall_acc_mean,
        'Overall_Accuracy_Std': overall_acc_std,
        'Overall_F1_Mean': overall_f1_mean,
        'Overall_F1_Std': overall_f1_std,
        'Split_Method': 'Random',
        'Num_Clients': num_clients
    }])
    overall_file = os.path.join(result_dir, f'hetefed_deap_{emotion}_overall_results_6clients_random_split.csv')
    overall_df.to_csv(overall_file, index=False)
    print(f"Overall results saved to: {overall_file}")

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")