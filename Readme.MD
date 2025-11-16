# Crypto Backtesting Framework

A professional-grade cryptocurrency backtesting and strategy optimization framework with ML integration.

## Features

- 📊 Historical backtesting for crypto strategies
- 🤖 ML-powered parameter optimization
- 📈 Institutional-grade performance metrics
- 🔄 Walk-forward analysis to prevent overfitting
- 🎯 Mean reversion strategies (extendable)
- 💾 Fast time-series storage with QuestDB

## Tech Stack

- **Python 3.10+**
- **QuestDB** - Time-series database
- **CCXT** - Exchange connectivity
- **Pandas/NumPy** - Data processing
- **Optuna** - Bayesian optimization
- **scikit-learn** - Machine learning

## Setup

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Poetry

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/crypto-backtest-framework.git
cd crypto-backtest-framework
```

2. Install dependencies:
```bash
poetry install
```

3. Start QuestDB:
```bash
docker-compose up -d
```

4. Copy environment file:
```bash
cp .env.example .env
```

5. Verify setup:
```bash
poetry run python -c "import pandas; print('✅ Setup complete!')"
```

## Project Structure
```
crypto-backtest-framework/
├── src/
│   ├── data/          # Data ingestion and storage
│   ├── indicators/    # Technical indicators
│   ├── strategies/    # Trading strategies
│   ├── backtesting/   # Backtesting engines
│   ├── optimization/  # ML optimization
│   └── analytics/     # Performance analysis
├── tests/             # Test suite
├── notebooks/         # Jupyter notebooks
├── config/            # Configuration files
└── data/              # Local data storage
```

## Usage

Coming soon...

## Development Roadmap

- [x] Phase 0: Project setup
- [ ] Phase 1: Data infrastructure
- [ ] Phase 2: Backtesting engine
- [ ] Phase 3: Strategy framework
- [ ] Phase 4: ML optimization
- [ ] Phase 5: Analytics & reporting

## License

MIT