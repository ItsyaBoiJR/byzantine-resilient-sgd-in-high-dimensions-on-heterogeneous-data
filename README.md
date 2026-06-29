# Byzantine-Resilient SGD in High Dimensions on Heterogeneous Data

## Introduction

This repository provides an implementation of the algorithm proposed in the research paper ["Byzantine-Resilient SGD in High Dimensions on Heterogeneous Data"](https://arxiv.org/pdf/2005.07866v1) by Deepesh Data and Suhas Diggavi. The paper addresses the challenge of performing distributed stochastic gradient descent (SGD) in a master-worker architecture in the presence of Byzantine failures (adversarial workers) and heterogeneous data across workers.

The proposed algorithm enhances the robustness of distributed training by filtering out corrupt gradients using a robust mean estimation procedure and analyzing the convergence properties of the method under various settings (convex and non-convex objectives). Additionally, an extension with gradient compression is introduced to reduce communication overhead while maintaining robustness.

## Core Concept

### Problem Setting
- **Distributed SGD**: A master-worker setup where workers compute gradients from their local datasets, and the master aggregates these gradients to perform model updates.
- **Byzantine Workers**: A fraction of workers may behave adversarially, sending malicious gradients to disrupt the training process.
- **Heterogeneous Data**: The data distribution across workers is non-i.i.d., with no probabilistic assumptions on how the data is generated.

### Proposed Solution
1. **Robust Gradient Aggregation**:
   - Leverages the outlier-filtering procedure from robust mean estimation literature (Steinhardt et al., ITCS 2018) to filter out malicious gradients.
   - Extends this procedure to handle stochastic gradients computed on heterogeneous data by deriving a novel matrix concentration result.
2. **Convergence Guarantees**:
   - Strongly-convex objectives: The algorithm achieves exponential convergence to an approximate optimal solution.
   - Non-convex objectives: The algorithm converges linearly to an approximate stationary point.
3. **Gradient Compression**:
   - A variant of the algorithm compresses gradients by selecting a random subset of coordinates, significantly reducing communication cost while retaining convergence guarantees.

### Key Features
- Tolerates up to 25% Byzantine workers.
- Matches the convergence rates of standard SGD in a Byzantine-free setting.
- Provides a trade-off between mini-batch size and approximation error.
- Offers a communication-efficient variant with gradient compression, saving up to a `d/k` factor in communication bits, where `d` is the dimension and `k` is the number of selected coordinates.

## Repository Overview

This repository contains a Python implementation of the Byzantine-resilient SGD algorithm using PyTorch. The implementation includes both the robust gradient aggregation procedure and the gradient compression variant.

### File Structure
```
.
├── README.md                # Documentation
├── byzantine_sgd.py         # Core implementation of the robust SGD algorithm
├── gradient_compression.py  # Implementation of the gradient compression variant
├── data_generator.py        # Utility to simulate heterogeneous and Byzantine data
├── experiments.py           # Scripts to run experiments and evaluate performance
├── requirements.txt         # Dependency list
└── utils.py                 # Helper functions for gradient aggregation and metrics
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your_username/byzantine-resilient-sgd.git
   cd byzantine-resilient-sgd
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Standard Byzantine-Resilient SGD
To run the robust SGD algorithm on a synthetic dataset:
```bash
python experiments.py --algorithm robust_sgd --dataset synthetic --num_workers 10 --byzantine_fraction 0.25
```

### Running the Gradient Compression Variant
To run the gradient compression variant:
```bash
python experiments.py --algorithm compressed_sgd --dataset synthetic --num_workers 10 --byzantine_fraction 0.25 --compression_rate 0.1
```

### Parameters
- `--algorithm`: Choose between `robust_sgd` and `compressed_sgd`.
- `--dataset`: Specify the dataset (`synthetic`, `mnist`, etc.).
- `--num_workers`: Number of workers in the setup.
- `--byzantine_fraction`: Fraction of workers that are Byzantine (e.g., 0.25 for 25%).
- `--compression_rate`: Fraction of gradient coordinates to retain (only applicable for `compressed_sgd`).

### Example Output
After running an experiment, the script will output:
- Training loss and accuracy over epochs.
- Robustness comparison with vanilla SGD under Byzantine attacks.
- Communication savings for the gradient compression algorithm.

## Code Description

### Core Components

1. **`byzantine_sgd.py`**:
   - Implements the robust gradient aggregation procedure using outlier filtering.
   - Handles both strongly-convex and non-convex optimization scenarios.
   - Incorporates matrix concentration bounds to adapt to heterogeneous data.

2. **`gradient_compression.py`**:
   - Implements the gradient compression algorithm by randomly selecting `k` coordinates from the gradient.
   - Reduces communication cost without significantly affecting convergence.

3. **`data_generator.py`**:
   - Simulates heterogeneous datasets and generates Byzantine workers’ malicious gradients.

4. **`experiments.py`**:
   - Runs experiments to evaluate the performance of the proposed algorithms under various settings.

5. **`utils.py`**:
   - Contains helper functions for gradient aggregation, computing metrics, and logging results.

## Results and Benchmarks

### Key Results from the Paper
- The robust SGD algorithm achieves Byzantine resilience with convergence rates comparable to standard SGD in non-adversarial settings.
- The gradient compression variant reduces communication overhead by a factor of `d/k`, where `d` is the gradient dimension and `k` is the number of selected coordinates, without degrading convergence.

### Experimental Results
The experiments demonstrate:
- Superior performance of the robust SGD algorithm in the presence of Byzantine workers.
- Significant communication savings for the gradient compression variant.

## Contributing

Contributions are welcome! If you encounter any issues, please open an issue or submit a pull request.

### Guidelines
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## References

1. Deepesh Data, Suhas Diggavi. "Byzantine-Resilient SGD in High Dimensions on Heterogeneous Data". [arXiv:2005.07866](https://arxiv.org/pdf/2005.07866v1)
2. Steinhardt, Jacob, et al. "Robust Learning: Information Theoretic Limits and Efficient Algorithms." ITCS 2018.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.