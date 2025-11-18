"""
Configuration loader for the backtesting framework.
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_name: Name of the config file (without .yaml extension)
        
    Returns:
        Dictionary containing configuration
    """
    config_path = PROJECT_ROOT / "config" / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_env(key: str, default: Any = None) -> str:
    """
    Get environment variable value.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value
    """
    return os.getenv(key, default)


# Database configuration
QUESTDB_CONFIG = {
    'host': get_env('QUESTDB_HOST', 'localhost'),
    'http_port': int(get_env('QUESTDB_HTTP_PORT', 9000)),
    'pg_port': int(get_env('QUESTDB_PG_PORT', 8812)),
    'ilp_port': int(get_env('QUESTDB_ILP_PORT', 9009)),
    'user': 'admin',
    'password': 'quest',
    'database': 'qdb'
}

# Backtesting configuration
BACKTEST_CONFIG = {
    'initial_capital': float(get_env('INITIAL_CAPITAL', 500)),
    'commission': float(get_env('DEFAULT_COMMISSION', 0.001)),
    'slippage': float(get_env('DEFAULT_SLIPPAGE', 0.0005))
}

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"\nQuestDB Config: {QUESTDB_CONFIG}")
    print(f"\nBacktest Config: {BACKTEST_CONFIG}")
    
    # Test loading YAML configs
    exchanges = load_config('exchanges')
    print(f"\nExchanges Config: {list(exchanges.keys())}")
    print(f"Configured Symbols: {exchanges['symbols']}")
