Use AutoRun.py to run the FedPhoenix algorithm and the Baselines algorithm

Sure, I can help you with that. Here's an MD file that explains the usage of the provided script:
# Experiment Configuration and Execution

## Introduction
This script is used to configure and run various experiments for federated learning algorithms. It allows you to specify different datasets, models, algorithms, and various training-related parameters.

## Usage
1. **Dataset Related Parameters**:
   - The `datasets` list specifies the datasets to be used for the experiments.
   - The `dataset_classes` dictionary defines the number of classes for each dataset.

2. **Algorithm Related Parameters**:
   - The `algorithms` list specifies the federated learning algorithms to be used.

3. **Model Related Parameters**:
   - The `models` list specifies the models to be used for the experiments.

4. **Training Related Parameters**:
   - `lrs`: Learning rates to be used.
   - `epochs`: Number of training epochs.
   - `data_betas`: Data distribution parameters, where 0.5 indicates IID data and other values indicate non-IID data.
   - `weight_decays`: Weight decay values.

5. **Federated Learning Related Parameters**:
   - `num_users`: Total number of clients.
   - `frac`: Fraction of clients participating in each round.

6. **Specific Algorithm Parameters**:
   - `fp_convs`: The total number of rounds for resetting in the Convolutional Layer of the FedPhoenix algorithm.Corresponding to r_s in the paper
   - `resets`: FedPhoenix algorithm's reset rate parameter.Corresponding to θ in the paper

7. **Hardware Related Parameters**:
   - `gpus`: List of GPU IDs to be used for the experiments.

8. **Resource Management Parameters**:
   - `GPU_MEMORY_THRESHOLD`: GPU memory usage threshold.
   - `GPU_UTIL_THRESHOLD`: GPU utilization threshold.
   - `TASK_INIT_SLEEP`: Task initialization wait time.
   - `SCHEDULER_SLEEP_BUSY`: Scheduler busy polling interval.
   - `SCHEDULER_SLEEP_IDLE`: Scheduler idle polling interval.

## Parameter Validation
The script includes a `validate_params()` function that checks the validity of the parameter configuration, ensuring that the values are within the expected ranges.

## Parameter Combinations
The script uses `itertools.product()` to generate all possible combinations of the specified parameters. It also dynamically generates the `fixed_params` dictionary based on the `data_beta` parameter, indicating whether the data is IID or non-IID.

## GPU Manager
The script includes a `GPUManager` class that is responsible for monitoring and managing the status and task allocation of multiple GPUs. It provides methods to check GPU availability, get the best available GPU, and add/clean up running processes.

## Main Function
The `main()` function is the entry point of the script. It initializes the `GPUManager`, schedules tasks on available GPUs, and manages the running tasks.