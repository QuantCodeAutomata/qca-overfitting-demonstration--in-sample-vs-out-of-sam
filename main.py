#!/usr/bin/env python3
"""
Main execution script for all portfolio optimization experiments.

This script runs all four experiments sequentially and generates
comprehensive results including visualizations and metrics.
"""

import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import initialize_results_file
from src.exp_1_overfitting import run_experiment_1, save_experiment_1_results
from src.exp_2_factor_priors import run_experiment_2, save_experiment_2_results
from src.exp_3_stacking import run_experiment_3, save_experiment_3_results
from src.exp_4_api_validation import run_experiment_4, save_experiment_4_results


def main():
    """
    Run all experiments and save results.
    """
    print("\n" + "="*80)
    print(" " * 20 + "PORTFOLIO OPTIMIZATION EXPERIMENTS")
    print("="*80 + "\n")
    
    # Initialize results file
    print("Initializing results file...")
    initialize_results_file()
    
    # Track experiment results
    all_results = {}
    
    try:
        # Experiment 1: Overfitting Demonstration
        print("\n\n" + "#"*80)
        print("# STARTING EXPERIMENT 1")
        print("#"*80 + "\n")
        
        exp1_results = run_experiment_1()
        save_experiment_1_results(exp1_results)
        all_results['exp_1'] = exp1_results
        
        print("\n✓ Experiment 1 completed successfully\n")
        
    except Exception as e:
        print(f"\n✗ Experiment 1 failed with error: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        # Experiment 2: Factor Model Prior Integration
        print("\n\n" + "#"*80)
        print("# STARTING EXPERIMENT 2")
        print("#"*80 + "\n")
        
        exp2_results = run_experiment_2()
        save_experiment_2_results(exp2_results)
        all_results['exp_2'] = exp2_results
        
        print("\n✓ Experiment 2 completed successfully\n")
        
    except Exception as e:
        print(f"\n✗ Experiment 2 failed with error: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        # Experiment 3: Stacking Optimization
        print("\n\n" + "#"*80)
        print("# STARTING EXPERIMENT 3")
        print("#"*80 + "\n")
        
        exp3_results = run_experiment_3()
        save_experiment_3_results(exp3_results)
        all_results['exp_3'] = exp3_results
        
        print("\n✓ Experiment 3 completed successfully\n")
        
    except Exception as e:
        print(f"\n✗ Experiment 3 failed with error: {e}\n")
        import traceback
        traceback.print_exc()
    
    try:
        # Experiment 4: API Validation
        print("\n\n" + "#"*80)
        print("# STARTING EXPERIMENT 4")
        print("#"*80 + "\n")
        
        exp4_results = run_experiment_4()
        save_experiment_4_results(exp4_results)
        all_results['exp_4'] = exp4_results
        
        print("\n✓ Experiment 4 completed successfully\n")
        
    except Exception as e:
        print(f"\n✗ Experiment 4 failed with error: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print(" " * 30 + "FINAL SUMMARY")
    print("="*80 + "\n")
    
    print(f"Experiments completed: {len(all_results)}/4")
    print("\nCompleted experiments:")
    for exp_name in all_results.keys():
        print(f"  ✓ {exp_name}")
    
    print("\nResults and visualizations saved to: results/")
    print("  - results/RESULTS.md (comprehensive metrics)")
    print("  - results/exp_1_efficient_frontier.png")
    print("  - results/exp_2_factor_prior_comparison.png")
    print("  - results/exp_3_stacking_comparison.png")
    print("  - results/exp_4_api_validation.png")
    
    print("\n" + "="*80)
    print(" " * 25 + "ALL EXPERIMENTS COMPLETED")
    print("="*80 + "\n")
    
    return all_results


if __name__ == "__main__":
    results = main()
    sys.exit(0)
