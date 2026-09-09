import sys
import os

# Add parent directory to sys.path so modules (geometry_engine, cad_exporter, app) resolve correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
