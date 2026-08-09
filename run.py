#!/usr/bin/env python3
"""
Main execution script for CNN-GRU Comparative Analysis project.
Runs signature recognition (CNN vs HOG vs SIFT) and LSTM text generation pipelines.
"""

import os
import sys
import subprocess
import argparse


def run_task1():
    """Execute signature recognition pipeline (CNN, HOG, SIFT)."""
    print("=" * 60)
    print("TASK 1: CNN FOR SIGNATURE RECOGNITION")
    print("=" * 60)

    try:
        os.chdir('task1_signature_recognition')
        result = subprocess.run([sys.executable, 'main_task1.py'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("Task 1 completed successfully!")
            print("Results saved to 'results/' directory")
        else:
            print("Error in Task 1:")
            print(result.stderr)

    except Exception as e:
        print(f"Error running Task 1: {e}")
    finally:
        os.chdir('..')


def run_task2():
    """Execute LSTM text generation pipeline."""
    print("
" + "=" * 60)
    print("TASK 2: LSTM FOR WORD COMPLETION")
    print("=" * 60)

    try:
        os.chdir('task2_word_completion')
        result = subprocess.run([sys.executable, 'main_task2.py'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("Task 2 completed successfully!")
            print("Results saved to 'results/' directory")
            print("
To run the Streamlit interface:")
            print("streamlit run streamlit_app.py")
        else:
            print("Error in Task 2:")
            print(result.stderr)

    except Exception as e:
        print(f"Error running Task 2: {e}")
    finally:
        os.chdir('..')


def run_streamlit():
    """Launch the Streamlit interface for text generation."""
    print("Launching Streamlit interface for word completion...")

    try:
        os.chdir('task2_word_completion')
        subprocess.run(['streamlit', 'run', 'streamlit_app.py'])
    except Exception as e:
        print(f"Error launching Streamlit: {e}")
    finally:
        os.chdir('..')


def main():
    parser = argparse.ArgumentParser(description='CNN-GRU Comparative Analysis')
    parser.add_argument('--task', choices=['1', '2', 'both', 'streamlit'],
                       default='both', help='Which task to run')
    parser.add_argument('--skip-task1', action='store_true',
                       help='Skip signature recognition')
    parser.add_argument('--skip-task2', action='store_true',
                       help='Skip text generation')

    args = parser.parse_args()

    print("CNN-GRU COMPARATIVE ANALYSIS")
    print("=" * 60)
    print("1. CNN for Signature Recognition with HOG/SIFT comparison")
    print("2. LSTM for Word Completion on Shakespeare dataset")
    print("=" * 60)

    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    if args.task == '1' or (args.task == 'both' and not args.skip_task1):
        run_task1()

    if args.task == '2' or (args.task == 'both' and not args.skip_task2):
        run_task2()

    if args.task == 'streamlit':
        run_streamlit()

    print("
" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("Check the 'results/' directory for all outputs")


if __name__ == "__main__":
    main()
