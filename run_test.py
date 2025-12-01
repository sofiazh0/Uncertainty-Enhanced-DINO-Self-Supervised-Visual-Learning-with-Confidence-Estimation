import os
import subprocess

def run_test():
    print("Running UA-DINO test...")
    
    # Verify all required files exist
    required_files = [
        'main_dino.py',
        'uncertainty_head.py',
        'uncertainty_loss.py',
        'uncertainty_metrics.py',
        'test_ua_dino.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: {file} not found!")
            return False
    
    # Run test
    try:
        subprocess.run(['python', 'test_ua_dino.py'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_test()
    print("Test completed successfully!" if success else "Test failed!")
