import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from config import Config
from model import PrunableMLP
from utils import set_seed, get_dataloaders, compute_sparsity, plot_gate_distribution

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

def run_experiment():
    cfg = Config()
    set_seed(cfg.seed)
    train_loader, test_loader = get_dataloaders(cfg.data_dir, cfg.batch_size)
    results = []

    for lmbda in cfg.lambdas:
        logging.info(f"Starting experiment with Lambda: {lmbda}")
        model = PrunableMLP().to(cfg.device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                ce_loss = criterion(outputs, targets)
                
                # L1 Penalty on gates
                reg_loss = sum([g.abs().sum() for g in model.get_all_gates()])
                loss = ce_loss + lmbda * reg_loss
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            if epoch % 5 == 0:
                logging.info(f"Epoch {epoch}/{cfg.epochs} - Loss: {train_loss/len(train_loader):.4f}")

        acc = evaluate(model, test_loader, cfg.device)
        sparsity = compute_sparsity(model, cfg.threshold)
        logging.info(f"Final Acc: {acc:.2f}% | Sparsity: {sparsity*100:.2f}%")
        
        # Save artifacts
        torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"model_lambda_{lmbda}.pt"))
        plot_gate_distribution(model, lmbda, cfg.output_dir)
        results.append((lmbda, acc, sparsity))

    # Print final table
    print("\n" + "="*40)
    print(f"| {'Lambda':<10} | {'Accuracy':<10} | {'Sparsity':<10} |")
    print("-"*40)
    for l, a, s in results:
        print(f"| {l:<10.1e} | {a:<10.2f} | {s*100:<10.2f} |")
    print("="*40)

    # Generate markdown table for report
    with open(os.path.join(cfg.report_dir, 'table.md'), 'w') as f:
        f.write("| Lambda | Accuracy | Sparsity |\n")
        f.write("| --- | --- | --- |\n")
        for l, a, s in results:
            f.write(f"| {l:.1e} | {a:.2f}% | {s*100:.2f}% |\n")

if __name__ == "__main__":
    run_experiment()
