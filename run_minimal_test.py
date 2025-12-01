import os
import sys
from test_ua_dino import main

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        sys.exit(1)
