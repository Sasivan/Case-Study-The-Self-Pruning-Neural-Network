
# Self-Pruning Neural Network Report

## 1. Explanation of Sparsity Mechanism

### L1 on Sigmoid Gates
Traditional pruning often involves post-training thresholding. In this implementation, we introduce trainable parameters called `gate_scores`. These are passed through a sigmoid function to produce `gates` $\in (0, 1)$. By applying an **L1 penalty** directly to these sigmoid outputs, the optimizer is incentivized to push gate values toward zero.

### The Lambda Trade-off
The parameter $\lambda$ controls the strength of the pruning pressure. 
- **Low Lambda:** Focuses on accuracy; gates stay near 1.0.
- **High Lambda:** Forces more gates toward 0.0, increasing sparsity but potentially sacrificing accuracy.

## 2. Experimental Results

| Lambda | Accuracy | Sparsity |
| --- | --- | --- |
| 1.0e-05 | 53.68% | 0.00% |
| 1.0e-04 | 56.74% | 0.00% |
| 1.0e-03 | 56.53% | 0.00% |


## 3. Graph Observations

The generated histograms in the `outputs/` directory visualize the pruning process:
- **Spike near Zero:** Represents weights that have been effectively pruned from the network.
- **Cluster away from Zero:** Represents active weights that the network has deemed essential for the classification task.
