from pathlib import Path

# Number pattern robust to -0.0000 and scientific notation
FLOAT = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'

__all__ = [
    "FLOAT",
]
