"""
Main script to run the entire Wide & Deep recommendation pipeline.
"""
import os
import argparse
import time
import yaml

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Wide & Deep Recommendation Pipeline')
    
    parser.add_argument('--preprocess', action='store_true',
                      help='Run data preprocessing step')
    parser.add_argument('--train', action='store_true',
                      help='Run model training step')
    parser.add_argument('--analyze', action='store_true',
                      help='Run model analysis step')
    parser.add_argument('--inference', action='store_true',
                      help='Run model inference step')
    parser.add_argument('--all', action='store_true',
                      help='Run all steps')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to configuration file')
    
    return parser.parse_args()

def run_preprocessing():
    """Run data preprocessing step."""
    print("\n" + "="*50)
    print("STEP 1: DATA PREPROCESSING")
    print("="*50)
    
    from scripts.data_processor import process_and_save_data
    
    start_time = time.time()
    processed_paths = process_and_save_data(overwrite=True)
    end_time = time.time()
    
    print(f"\nPreprocessing completed in {end_time - start_time:.2f} seconds")
    print(f"Processed data saved to:")
    for key, path in processed_paths.items():
        if path:
            print(f"  - {key}: {path}")
    
    return processed_paths

def run_training(config_path):
    """Run model training step."""
    print("\n" + "="*50)
    print("STEP 2: MODEL TRAINING")
    print("="*50)
    
    from scripts.train import train_and_visualize
    
    start_time = time.time()
    history, results, model = train_and_visualize()
    end_time = time.time()
    
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    print(f"Model saved to saved_model/wide_deep_model.keras")
    
    return model

def run_analysis():
    """Run model analysis step."""
    print("\n" + "="*50)
    print("STEP 3: MODEL ANALYSIS")
    print("="*50)
    
    # Check if model exists
    model_path = 'saved_model/wide_deep_model.keras'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run the training step first")
        return
    
    try:
        # Import here to avoid loading unnecessary modules
        from scripts.analyze import main as analyze_main
        
        start_time = time.time()
        analyze_main()
        end_time = time.time()
        
        print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")
        print(f"Results saved to analysis_results/")
    except ImportError:
        print("Error: Could not import analysis module")
        print("Make sure you have all required packages installed")

def run_inference():
    """Run model inference step."""
    print("\n" + "="*50)
    print("STEP 4: MODEL INFERENCE")
    print("="*50)
    
    # Check if model exists
    model_path = 'saved_model/wide_deep_model.keras'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run the training step first")
        return
    
    try:
        # Import here to avoid loading unnecessary modules
        from scripts.inference import main as inference_main
        
        start_time = time.time()
        inference_main()
        end_time = time.time()
        
        print(f"\nInference completed in {end_time - start_time:.2f} seconds")
        print(f"Results saved to inference_results/")
    except ImportError:
        print("Error: Could not import inference module")
        print("Make sure you have all required packages installed")

def main():
    """Main function to run the pipeline."""
    args = parse_arguments()
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('saved_model', exist_ok=True)
    os.makedirs('analysis_results', exist_ok=True)
    os.makedirs('inference_results', exist_ok=True)
    
    # If no specific steps are selected, show help
    if not (args.preprocess or args.train or args.analyze or args.inference or args.all):
        print("No steps selected. Please specify at least one step or use --all.")
        print("Run with --help for more information.")
        return
    
    # Run selected steps
    if args.preprocess or args.all:
        processed_paths = run_preprocessing()
        # Update config path if preprocessing was run
        if processed_paths and 'config' in processed_paths:
            args.config = processed_paths['config']
    
    if args.train or args.all:
        model = run_training(args.config)
    
    if args.analyze or args.all:
        run_analysis()
    
    if args.inference or args.all:
        run_inference()
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)
    print("\nSummary of outputs:")
    print("  - Processed data: data/processed/")
    print("  - Model: saved_model/wide_deep_model.keras")
    print("  - Training logs: logs/")
    print("  - Analysis results: analysis_results/")
    print("  - Inference results: inference_results/")
    print("\nTo visualize the results, check the PNG files in the logs/ directory.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error message above and try again.")