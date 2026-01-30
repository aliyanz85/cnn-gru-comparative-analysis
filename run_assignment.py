#!/usr/bin/env python3
"""
Main script to run Assignment #1: Generative AI
This script executes both Task 1 (CNN for Signature Recognition) and Task 2 (LSTM for Word Completion)
"""

import os
import sys
import subprocess
import argparse

def run_task1():
    """
    Execute Task 1: CNN for Signature Recognition
    """
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
    """
    Execute Task 2: LSTM for Word Completion
    """
    print("\n" + "=" * 60)
    print("TASK 2: LSTM FOR WORD COMPLETION")
    print("=" * 60)
    
    try:
        os.chdir('task2_word_completion')
        result = subprocess.run([sys.executable, 'main_task2.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Task 2 completed successfully!")
            print("Results saved to 'results/' directory")
            print("\nTo run the Streamlit interface:")
            print("streamlit run streamlit_app.py")
        else:
            print("Error in Task 2:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Error running Task 2: {e}")
    finally:
        os.chdir('..')

def run_streamlit():
    """
    Launch the Streamlit interface for Task 2
    """
    print("Launching Streamlit interface for word completion...")
    
    try:
        os.chdir('task2_word_completion')
        subprocess.run(['streamlit', 'run', 'streamlit_app.py'])
    except Exception as e:
        print(f"Error launching Streamlit: {e}")
    finally:
        os.chdir('..')

def main():
    """
    Main function to run the assignment
    """
    parser = argparse.ArgumentParser(description='Run Assignment #1: Generative AI')
    parser.add_argument('--task', choices=['1', '2', 'both', 'streamlit'], 
                       default='both', help='Which task to run')
    parser.add_argument('--skip-task1', action='store_true', 
                       help='Skip Task 1 (CNN for Signature Recognition)')
    parser.add_argument('--skip-task2', action='store_true', 
                       help='Skip Task 2 (LSTM for Word Completion)')
    
    args = parser.parse_args()
    
    print("ASSIGNMENT #1: GENERATIVE AI")
    print("=" * 60)
    print("This assignment implements:")
    print("1. CNN for Signature Recognition with HOG/SIFT comparison")
    print("2. LSTM for Word Completion on Shakespeare dataset")
    print("=" * 60)
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Warning: Virtual environment not detected.")
        print("Please activate the virtual environment first:")
        print("source gen_ai_env/bin/activate")
        print()
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if args.task == '1' or (args.task == 'both' and not args.skip_task1):
        run_task1()
    
    if args.task == '2' or (args.task == 'both' and not args.skip_task2):
        run_task2()
    
    if args.task == 'streamlit':
        run_streamlit()
    
    print("\n" + "=" * 60)
    print("ASSIGNMENT COMPLETED")
    print("=" * 60)
    print("Check the 'results/' directory for all outputs")
    print("Technical report is available in 'report/technical_report.tex'")
    print("\nTo create the final ZIP bundle, run:")
    print("python create_zip.py")

if __name__ == "__main__":
    main()
