#!/usr/bin/env python
"""
CLI wrapper for continuous learning
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continuous_learning.cli import main

if __name__ == "__main__":
    main()
