"""
Run multiple experiments with different Wide & Deep model configurations.
"""
import os
import sys
import time
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import shutil
from datetime import datetime
import tensorflow as tf

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Wide & Deep Recommendation Experiments')
    
    parser.add_argument('--experiment', type=str, default=None,
                      help='Name of a specific experiment to run')
    parser.add_argument('--run-all', action='store_true',
                      help='Run all predefined experiments')
    parser.add_argument('--create-configs', action='store_true',
                      help='Create configuration files without running experiments')
    parser.add_argument('--compare', action='store_true',
                      help='Compare results of already run experiments')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                      help='Directory for experiment outputs')
    parser.add_argument('--configs-dir', type=str, default='experiment_configs',
                      help='Directory for experiment configuration files')
    
    return parser.parse_args()

def create_experiment_configs(configs_dir):
    """Create configuration files for each experiment."""
    os.makedirs(configs_dir, exist_ok=True)
    
    # Load base configuration
    with open('configs/config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Define experiment variations
    experiments = {
        'baseline': {},  # Use default parameters
        
        'deep_network': {
            'model': {
                'hidden_units': [256, 256, 256],
                'dropout_rate': 0.5,
            }
        },
        
        'high_dropout': {
            'model': {
                'dropout_rate': 0.8,
            }
        },
        
        'no_dropout': {
            'model': {
                'dropout_rate': 0.0,
            }
        },
        
        'small_batch': {
            'train': {
                'batch_size': 32,
                'epochs': 20,
            }
        },
        
        'large_batch': {
            'train': {
                'batch_size': 1024,
                'epochs': 5,
            }
        },
        
        'high_learning_rate': {
            'train': {
                'learning_rate': 0.01,
            }
        },
        
        'low_learning_rate': {
            'train': {
                'learning_rate': 0.0001,
            }
        },
        
        'sgd_optimizer': {
            'train': {
                'optimizer': 'sgd',
                'learning_rate': 0.01,
                'momentum': 0.9,
            }
        },
        
        'wide_only': {
            'model': {
                'architecture': 'wide_only',
            }
        },
        
        'deep_only': {
            'model': {
                'architecture': 'deep_only',
            }
        },
        
        'large_embeddings': {
            'model': {
                'embedding_dim': 64,
            }
        },
        
        'minimal_epochs': {
            'train': {
                'epochs': 3,
            }
        },
        
        'comedy_only': {
            'data': {
                'genre_filter': 'genre_Comedy',
            }
        },
        
        'no_early_stopping': {
            'train': {
                'early_stopping': False,
                'epochs': 30,
            }
        },
    }
    
    # Create config files
    for exp_name, exp_params in experiments.items():
        # Start with base config
        exp_config = dict(base_config)
        
        # Add experiment name
        exp_config['experiment'] = {
            'name': exp_name,
        }
        
        # Apply experiment parameters
        for category, params in exp_params.items():
            if category not in exp_config:
                exp_config[category] = {}
            
            for param, value in params.items():
                exp_config[category][param] = value
        
        # Save config
        config_path = os.path.join(configs_dir, f"{exp_name}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(exp_config, f)
        
        print(f"Created experiment config: {config_path}")
    
    return list(experiments.keys())

def run_experiment(exp_name, exp_dir, config_path):
    """Run a single experiment."""
    print("\n" + "="*60)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print("="*60)
    
    # Create experiment directory
    os.makedirs(exp_dir, exist_ok=True)
    
    # Copy config to experiment directory
    shutil.copy(config_path, os.path.join(exp_dir, 'config.yaml'))
    
    # Set environment variables to redirect outputs
    os.environ['EXPERIMENT_DIR'] = exp_dir
    
    # Set TensorFlow session
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Run the experiment
    start_time = time.time()
    
    try:
        # Import at runtime to use the latest code
        from scripts.data_processor import process_and_save_data
        from scripts.train import train_and_visualize
        
        # Process data
        process_and_save_data(config_path, overwrite=True)
        
        # Train model
        history, results, model = train_and_visualize(config_path=config_path, output_dir=exp_dir)
        
        # Save results
        results_file = os.path.join(exp_dir, 'results.json')
        results_dict = {
            'loss': float(results[0]),
            'auc': float(results[1]),
            'precision': float(results[2]),
            'recall': float(results[3]),
            'runtime': time.time() - start_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Experiment {exp_name} completed in {time.time() - start_time:.2f} seconds")
        print(f"Results saved to {results_file}")
        
        return results_dict
    
    except Exception as e:
        print(f"Experiment {exp_name} failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        error_file = os.path.join(exp_dir, 'error.txt')
        with open(error_file, 'w') as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())
        
        return None

def compare_results(experiments_dir):
    """Compare results from multiple experiments."""
    # Find all experiment results
    results = {}
    for exp_name in os.listdir(experiments_dir):
        exp_dir = os.path.join(experiments_dir, exp_name)
        results_file = os.path.join(exp_dir, 'results.json')
        
        if os.path.isdir(exp_dir) and os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results[exp_name] = json.load(f)
            except:
                print(f"Could not load results for {exp_name}")
    
    if not results:
        print("No experiment results found.")
        return
    
    # Create results DataFrame
    df = pd.DataFrame(results).T
    
    # Sort by AUC
    df = df.sort_values('auc', ascending=False)
    
    # Create comparison directory
    comparison_dir = os.path.join(experiments_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Save results to CSV
    csv_path = os.path.join(comparison_dir, 'results_comparison.csv')
    df.to_csv(csv_path)
    
    # Plot metrics comparison
    metrics = ['auc', 'precision', 'recall', 'loss']
    plt.figure(figsize=(12, 8))
    df[metrics].plot(kind='bar', figsize=(12, 8))
    plt.title('Experiment Results Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()
    
    # Plot runtime comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df.index, y='runtime', data=df)
    plt.title('Experiment Runtime Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Runtime (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'runtime_comparison.png'), dpi=300)
    plt.close()
    
    # Print results table
    print("\nExperiment Results:")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(df[['auc', 'precision', 'recall', 'loss', 'runtime']])
    print("\nResults saved to:", comparison_dir)

def main():
    """Main function to run experiments."""
    args = parse_arguments()
    
    # Ensure directories exist
    configs_dir = args.configs_dir
    experiments_dir = args.experiments_dir
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Create experiment configs
    experiment_names = create_experiment_configs(configs_dir)
    
    # If only creating configs, exit
    if args.create_configs:
        print(f"Created {len(experiment_names)} experiment configurations in {configs_dir}/")
        return
    
    # If comparing results, run comparison and exit
    if args.compare:
        compare_results(experiments_dir)
        return
    
    # Run specific experiment
    if args.experiment:
        if args.experiment not in experiment_names:
            print(f"Error: Experiment '{args.experiment}' not found. Available experiments:")
            for name in experiment_names:
                print(f"  - {name}")
            return
        
        exp_name = args.experiment
        config_path = os.path.join(configs_dir, f"{exp_name}.yaml")
        exp_dir = os.path.join(experiments_dir, exp_name)
        
        run_experiment(exp_name, exp_dir, config_path)
        
        # Compare results
        compare_results(experiments_dir)
        return
    
    # Run all experiments
    if args.run_all:
        all_results = {}
        
        for exp_name in experiment_names:
            config_path = os.path.join(configs_dir, f"{exp_name}.yaml")
            exp_dir = os.path.join(experiments_dir, exp_name)
            
            results = run_experiment(exp_name, exp_dir, config_path)
            if results:
                all_results[exp_name] = results
        
        # Compare results
        compare_results(experiments_dir)
        return
    
    # If no action specified, print help
    print("No action specified. Use one of the following arguments:")
    print("  --experiment NAME  Run a specific experiment")
    print("  --run-all          Run all predefined experiments")
    print("  --create-configs   Create configuration files without running experiments")
    print("  --compare          Compare results of already run experiments")
    print("\nRun with --help for more information.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExperiments interrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()