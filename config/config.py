# config/config.py
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define paths to specific directories relative to the project root
NETGEN_DIR = os.path.join(PROJECT_ROOT, 'netgen')
TESTS_DIR = os.path.join(PROJECT_ROOT, 'tests')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
