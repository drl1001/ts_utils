#!/usr/bin/env python3

import matplotlib.pyplot as plt
import re
import sys
import os
import numpy as np

def plot_steady_conv(filename: str):
    """Plot convergence for steady-state simulations"""
    # Read the log file
    with open(filename, 'r') as file:
        log_content = file.read()

    # Extract ROVX DELTA values and iteration numbers
    iterations = []
    rovx_delta_values = []

    # Pattern to match RK LOOP NO and ROVX DELTA
    pattern = r'RK LOOP NO:\s+(\d+).*?ROVX DELTA:\s+([\d\.\-eE]+)'
    matches = re.findall(pattern, log_content, re.DOTALL)

    for match in matches:
        iteration = int(match[0])
        rovx_delta = float(match[1])
        iterations.append(iteration)
        rovx_delta_values.append(rovx_delta)

    # Plot the convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(iterations, rovx_delta_values, 'bo-', linewidth=2, markersize=3)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('ROVX DELTA', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(f'Steady Convergence', fontsize=24)
    plt.grid(True, which="both", ls="-", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print extracted values summary
    if iterations:
        print(f"Steady analysis completed for {filename}")
        print(f"Total iterations: {len(iterations)}")
        print(f"Final ROVX DELTA: {rovx_delta_values[-1]:.2e}")
    else:
        print("Warning: No RK LOOP data found in log file")

def parse_turbostream_log(filename):
    """
    Parse TurboStream log file and extract convergence data for outer steps
    """
    # Patterns to match outer step data
    outer_step_pattern = r'OUTER STEP NO\. (\d+)'
    residual_patterns = {
        'RO': r'RO RESIDUAL: ([\d\.eE+-]+)',
        'ROVX': r'ROVX RESIDUAL: ([\d\.eE+-]+)', 
        'ROVY': r'ROVY RESIDUAL: ([\d\.eE+-]+)',
        'ROVZ': r'ROVZ RESIDUAL: ([\d\.eE+-]+)',
        'ROE': r'ROE RESIDUAL: ([\d\.eE+-]+)',
        'ROVX_DELTA': r'ROVX DELTA:\s+([\d\.eE+-]+)'
    }
    
    # Storage for data
    outer_steps = []
    residuals = {key: [] for key in residual_patterns.keys()}
    
    with open(filename, 'r') as file:
        content = file.read()
    
    # Split content by outer steps
    outer_step_sections = re.split(r'OUTER STEP NO\. \d+', content)
    
    # The first section is before any outer steps, so we skip it
    for i, section in enumerate(outer_step_sections[1:], 1):
        outer_steps.append(i-1)  # Start from step 0
        
        # Extract residuals for this outer step
        for key, pattern in residual_patterns.items():
            match = re.search(pattern, section)
            if match:
                value = float(match.group(1))
                residuals[key].append(value)
            else:
                # If not found, use the previous value or NaN
                if len(residuals[key]) > 0:
                    residuals[key].append(residuals[key][-1])
                else:
                    residuals[key].append(float('nan'))
    
    return outer_steps, residuals

def plot_unsteady_conv(filename: str):
    """Plot convergence for unsteady simulations"""
    try:
        # Parse the log file
        print(f"Parsing log file: {filename}")
        outer_steps, residuals = parse_turbostream_log(filename)
        
        # Print some statistics
        print(f"Found {len(outer_steps)} outer steps")
        print("\nResidual ranges:")
        for key, values in residuals.items():
            if values:
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    print(f"  {key}: {min(valid_values):.2e} to {max(valid_values):.2e}")
        
        # Plot the convergence
        fig = plt.figure(figsize=(12, 10))
        
        if 'ROVX_DELTA' in residuals and len(residuals['ROVX_DELTA']) > 0:
            plt.semilogy(outer_steps, residuals['ROVX_DELTA'], 
                        label='ROVX DELTA', 
                        color='black',
                        linewidth=2,
                        marker='s' if len(outer_steps) < 50 else '',
                        markersize=4)
        
        plt.xlabel('Outer Step', fontsize=20)
        plt.ylabel('ROVX DELTA Value', fontsize=20)
        plt.title(f'Unsteady Convergence', fontsize=24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    """Main function to handle command line arguments"""
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python plot_convergence.py <log_file> <steady|unsteady>")
        print("Example: python plot_convergence.py log_2.txt steady")
        sys.exit(1)
    
    # Get arguments
    log_file = sys.argv[1]
    simulation_type = sys.argv[2].lower()
    
    # Check if file exists
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        sys.exit(1)
    
    # Choose plotting function based on simulation type
    if simulation_type == 'steady':
        plot_steady_conv(log_file)
    elif simulation_type == 'unsteady':
        plot_unsteady_conv(log_file)
    else:
        print(f"Error: Unknown simulation type '{simulation_type}'. Use 'steady' or 'unsteady'.")
        sys.exit(1)

if __name__ == "__main__":
    main()