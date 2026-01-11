---
title: Crypto Backtesting Framework - Complete Execution Guide
version: 3.0
created: 2025-11-30
purpose: Master guide for Claude Code to execute full migration and deployment
status: READY FOR EXECUTION
---

# 🚀 COMPLETE EXECUTION GUIDE
## Mean Reversion Trading Framework with Order Book Data

This is the **master guide** for executing the complete migration from QuestDB to TimescaleDB, integrating Crypto Lake order book data, and deploying production mean reversion trading strategies.

---

## 📚 DOCUMENTATION STRUCTURE

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **THIS FILE** | Overview, quick start, phase checklist | Start here |
| `MIGRATION_EXECUTION_PLAN.md` | Detailed Phase 0-1 implementation | During execution |
| `PHASE_GUIDES/` | Individual phase guides (2-15) | Phase-by-phase execution |
| `CLAUDE.md` | Framework overview & daily commands | Reference |

---

## 🎯 MISSION & GOALS

### Primary Objective
Build a production-grade cryptocurrency trading system that generates **consistent alpha** through:
- Mean reversion strategies using order book microstructure
- High-frequency features (VPIN, imbalance, queue dynamics)
- Dynamic transaction cost modeling
- Regime-aware execution

### Performance Targets

| Metric | Target (Conservative) | Target (Realistic) |
|--------|---------------------|-------------------|
| **Out-of-Sample Sharpe** | 1.0-1.2 | 1.5-2.0 |
| **Annual Return** | 20-30% | 40-60% |
| **Max Drawdown** | <25% | <18% |
| **Win Rate** | >50% | >53% |
| **Avg Hold Time** | 2-4 hours | 45-90 min |

### Success Metrics
- ✅ Strategy profitable with **2x transaction costs**
- ✅ Works across **multiple symbols** (BTC, SOL, ETH)
- ✅ Robust across **volatility regimes**
- ✅ Sharpe >1.0 in **paper trading** for 30 days

---

## 📋 15-PHASE EXECUTION PLAN

### PHASE 0: Proof of Concept (1 Week) - MANDATORY FIRST
**Status:** Required before full migration
**Duration:** 5-7 days
**Objective:** Validate that order book features improve predictions

**Tasks:**
1. Set up minimal TimescaleDB with order book deltas table
2. Fetch 7 days of BTC/USDT data from Crypto Lake
3. Engineer 5-10 core features (imbalance, VPIN, zscore)
4. Train simple LightGBM model
5. Backtest with realistic transaction costs
6. **GO/NO-GO Decision**

**Success Criteria:**
- Test correlation with returns >0.05
- Backtest Sharpe >1.0 (before costs)
- Profitable with 2x transaction costs
- TimescaleDB queries <1s for 1-day windows

**→ See:** `MIGRATION_EXECUTION_PLAN.md` (Lines 1-1200) for full POC implementation

**Commit Template:**
```
[POC] Complete proof of concept - GO/NO-GO decision

Results:
- Sharpe (gross): X.XX
- Sharpe (2x costs): X.XX
- Top features: [list]
- Decision: GO / NO-GO

Next: PHASE 1 if GO
```

---

### PHASE 1: Environment Setup (2-3 Days)
**Prerequisites:** POC passed
**Objective:** Production TimescaleDB + dependencies

**Key Deliverables:**
- `docker-compose.yml` with optimized TimescaleDB
- Updated `pyproject.toml` (lake-api, shap, boruta, mlflow)
- Project structure (features/, execution/, validation/)
- Environment configuration (.env)

**Verification:**
```bash
docker-compose up -d
poetry install
psql -h localhost -U postgres -d crypto_backtest -c "SELECT version();"
python -c "import lake, shap; print('✅')"
```

**→ See:** `MIGRATION_EXECUTION_PLAN.md` (Lines 1201-1600)

---

### PHASE 2: TimescaleDB Storage Layer (3-4 Days)
**Prerequisites:** Phase 1 complete
**Objective:** Create production schema with order book deltas

**Key Deliverables:**
- `src/data/storage/timescaledb.py` - Storage class
- SQL schema (order_book_deltas, trades, liquidations, funding_rates)
- Continuous aggregates (1-min snapshots, trade flow)
- Migration script from QuestDB (if needed)

**Critical Tables:**
1. `order_book_deltas` - Event-based storage (not snapshots)
2. `trades` - With aggressor classification
3. `liquidations` - For cascade prediction
4. `funding_rates` - For arbitrage opportunities
5. `ohlcv` - Existing candle data

**Verification:**
```bash
python scripts/test_timescaledb_storage.py
# Should insert 1M rows in <30s, query 1-day in <1s
```

**→ Create:** `docs/PHASE_02_TIMESCALEDB.md` with detailed implementation

---

### PHASE 3: Crypto Lake Integration (4-5 Days)
**Prerequisites:** Phase 2 complete
**Objective:** Download historical order book deltas + trades

**Data Collection Plan:**

| Asset | Data Type | Start Date | Priority | Est. Size |
|-------|-----------|------------|----------|-----------|
| BTC/USDT | book_delta_v2 | 2022-11-14 | P0 | ~800 MB |
| BTC/USDT | trades | 2020-01-01 | P0 | ~500 MB |
| BTC/USDT-PERP | book_delta_v2 | 2022-11-14 | P0 | ~800 MB |
| BTC/USDT-PERP | funding | 2020-01-01 | P0 | ~10 MB |
| BTC/USDT-PERP | liquidations | 2020-01-01 | P1 | ~50 MB |
| SOL/USDT | book_delta_v2 | 2022-11-14 | P1 | ~400 MB |

**Key Deliverables:**
- `src/data/providers/cryptolake_provider.py`
- `scripts/download_historical_data.py`
- Aggressor classification logic
- Data quality validation

**Verification:**
```bash
python scripts/download_historical_data.py --symbol BTC/USDT --start 2022-11-14
# Should download and validate data with <1% gaps
```

**→ Create:** `docs/PHASE_03_CRYPTO_LAKE.md`

---

### PHASE 4: Data Validation (2-3 Days)
**Prerequisites:** Phase 3 complete
**Objective:** Pandera schemas + quality checks

**Key Deliverables:**
- `src/data/validators/schemas.py` - Extended schemas
- Order book delta validation (no crossed spreads)
- Trade aggressor validation (all classified)
- Gap detection and reporting

**Validation Rules:**
- No negative prices/volumes
- No crossed spreads (bid < ask)
- Timestamps monotonic increasing
- All trades have aggressor classification
- Max gap <10 minutes

**→ Create:** `docs/PHASE_04_VALIDATION.md`

---

### PHASE 5: Data Loaders (2 Days)
**Prerequisites:** Phase 4 complete
**Objective:** Unified data loading interface

**Key Deliverables:**
- `src/data/loaders/data_loader.py` - Updated for TimescaleDB
- Support for order book, trades, OHLCV, funding
- Efficient joins across data types
- Caching strategy

**API:**
```python
loader = DataLoader(exchange='binance')

# Load OHLCV
ohlcv = loader.load_ohlcv('BTC/USDT', '1m', start, end)

# Load order book deltas (reconstructed to snapshots)
order_book = loader.load_order_book_snapshots('BTC/USDT', start, end, interval='1m')

# Load trades with aggressors
trades = loader.load_trades('BTC/USDT', start, end)

# Load combined dataset for ML
combined = loader.load_combined('BTC/USDT', '1m', start, end)
```

**→ Create:** `docs/PHASE_05_DATA_LOADERS.md`

---

### PHASE 6: Feature Engineering (7-9 Days) - **CRITICAL FOR ALPHA**
**Prerequisites:** Phase 5 complete
**Objective:** Build 50+ features for mean reversion

**Feature Categories:**

#### 6.1 Order Book Microstructure (15-20 features)
- Depth imbalance (5, 10, 20 levels)
- VPIN (order flow toxicity)
- Queue dynamics (spoofing detection)
- Liquidity fragility (Kyle's lambda)
- Spread metrics

#### 6.2 Mean Reversion Signals (10-15 features)
- **Half-life estimation** (Ornstein-Uhlenbeck process)
- Price z-scores (multiple lookbacks)
- Deviation from fair values (VWAP, funding parity, EMA)
- Bollinger Band positions
- RSI oversold/overbought
- Hurst exponent (mean reversion vs trending)

#### 6.3 Volatility & Regime (10-12 features)
- Realized kernel volatility
- Jump detection (Barndorff-Nielsen)
- Volatility regime classification
- ADX (trend strength - avoid MR in strong trends)

#### 6.4 Cross-Asset Features (8-10 features)
- BTC lead-lag signals (for altcoins)
- Cross-asset correlations
- Relative strength (SOL/BTC ratio)

#### 6.5 Funding & Perpetuals (8-10 features)
- Funding rate z-scores
- Spot-perp basis
- Time to funding
- Liquidation risk scores

#### 6.6 Temporal Features (6-8 features)
- Hour of day (US/Asia trading hours)
- Day of week
- Month-end effects
- Pre-funding flows

#### 6.7 Nonlinear & Interactions (10-15 features)
- Imbalance × Volatility
- VPIN × Trade Size
- Spread × Imbalance
- Polynomial price z-scores

**Key Files to Create:**
- `src/features/microstructure.py`
- `src/features/mean_reversion.py` ← **MOST IMPORTANT**
- `src/features/volatility.py`
- `src/features/cross_asset.py`
- `src/features/funding.py`
- `src/features/temporal.py`
- `src/features/nonlinear.py`

**Verification:**
```python
from src.features.mean_reversion import calculate_half_life, calculate_fair_value_deviations

# Test on BTC data
half_life = calculate_half_life(price_series)
print(f"Mean reversion half-life: {half_life} minutes")  # Should be 15-60 min

# Top features should correlate with forward returns
correlations = analyze_feature_correlations(features, forward_returns)
# Expect top 10 features to have |corr| > 0.03
```

**→ Create:** `docs/PHASE_06_FEATURES.md` (most detailed guide)

---

### PHASE 7: Configuration (1-2 Days)
**Prerequisites:** Phase 6 complete
**Objective:** YAML configs for backtesting, ML, and production

**Key Files:**
- `config/timescaledb.yaml` - Database settings
- `config/exchanges.yaml` - Updated with Crypto Lake
- `config/backtesting.yaml` - Dynamic transaction costs
- `config/ml_optimization.yaml` - LightGBM + validation settings

**Critical Config Additions:**
```yaml
# config/backtesting.yaml

transaction_costs:
  model: dynamic  # Not static!

  # Commission (tier-based)
  maker_fee_bps: 2
  taker_fee_bps: 5

  # Market impact
  impact_model: almgren_chriss
  kyle_lambda: auto  # Estimated from data

  # Adverse selection
  adverse_selection_model: vpin_based

# Mean reversion specific
mean_reversion:
  min_half_life_minutes: 15
  max_half_life_minutes: 240
  min_zscore_threshold: 1.5
  regime_filter: true  # Don't trade in strong trends
```

**→ Create:** `docs/PHASE_07_CONFIG.md`

---

### PHASE 8: Testing & Documentation (3-4 Days)
**Prerequisites:** Phase 7 complete
**Objective:** Comprehensive test suite + updated docs

**Test Coverage:**
- Unit tests for all feature calculations
- Integration tests (TimescaleDB + Crypto Lake)
- Feature importance tests (SHAP validation)
- Transaction cost sensitivity tests

**Documentation Updates:**
- Update `CLAUDE.md` with new commands
- Create feature engineering guide
- API documentation
- Troubleshooting guide

**→ Create:** `docs/PHASE_08_TESTING.md`

---

### PHASE 9: ML Optimization (4-5 Days)
**Prerequisites:** Phase 8 complete
**Objective:** LightGBM training with overfitting prevention

**Key Components:**

#### 9.1 Feature Selection
- Variance threshold
- Correlation filtering (remove >0.95 corr)
- Boruta feature selection
- SHAP-based importance
- **Target:** 30-50 features (from 100+)

#### 9.2 Model Training
- LightGBM with conservative hyperparameters
- Early stopping
- Cross-validation (purged k-fold)

#### 9.3 Validation Framework
- Expanding window walk-forward
- Cross-symbol validation (train BTC, test SOL)
- Regime-aware splitting
- Transaction cost sensitivity

**Key Files:**
- `src/optimization/ml/feature_selector.py`
- `src/optimization/ml/model_trainer.py`
- `src/validation/walk_forward.py`
- `src/validation/cross_symbol.py`

**→ Create:** `docs/PHASE_09_ML_OPTIMIZATION.md`

---

### PHASE 10: Mean Reversion Strategies (5-6 Days)
**Prerequisites:** Phase 9 complete
**Objective:** Implement dedicated MR strategies

**Strategy Types:**

#### 10.1 Statistical Mean Reversion
- Z-score based entry/exit
- Half-life optimized holding periods
- Regime-filtered execution

#### 10.2 Funding Arbitrage
- Spot-perp basis trades
- Pre-funding flow exploitation
- Extreme funding mean reversion

#### 10.3 Microstructure Mean Reversion
- Imbalance overshoots
- VPIN-filtered entry
- Queue position opportunities

**Key Features:**
- Passive vs aggressive execution modes
- Position sizing based on half-life
- Stop-loss at 2× expected reversion
- Regime kill-switch (no MR in strong trends)

**Key Files:**
- `src/strategies/mean_reversion/statistical_mr.py`
- `src/strategies/mean_reversion/funding_arbitrage.py`
- `src/strategies/mean_reversion/microstructure_mr.py`

**→ Create:** `docs/PHASE_10_STRATEGIES.md`

---

### PHASE 11: Transaction Cost Modeling (3-4 Days)
**Prerequisites:** Phase 10 complete
**Objective:** Dynamic cost model for realistic backtesting

**Components:**
1. Commission (maker/taker, tier-based)
2. Spread cost (dynamic based on market conditions)
3. Market impact (Almgren-Chriss temporary + Kyle permanent)
4. Adverse selection (VPIN-based)
5. Opportunity cost (for passive orders)

**Key File:**
- `src/execution/cost_models/dynamic_cost_model.py`

**Verification:**
```python
cost_model = DynamicTransactionCostModel()

# Estimate cost for 1% ADV trade
cost = cost_model.estimate_cost(
    symbol='BTC/USDT',
    quantity_usd=100000,
    urgency='aggressive',
    market_state={
        'spread_bps': 5,
        'vpin': 0.4,
        'kyle_lambda': 0.0001,
        'realized_vol': 0.6
    }
)

print(f"Expected cost: {cost['expected_cost_bps']:.1f} bps")
# Should be 10-30 bps for aggressive, 2-10 bps for passive
```

**→ Create:** `docs/PHASE_11_TRANSACTION_COSTS.md`

---

### PHASE 12: Validation Framework (4-5 Days)
**Prerequisites:** Phase 11 complete
**Objective:** Rigorous out-of-sample validation

**Validation Layers:**

#### 12.1 Expanding Window Walk-Forward
- Min 90-day train, 30-day test
- Step size: 30 days
- Purge: 1-day embargo

#### 12.2 Cross-Symbol Validation
- Train on BTC → Test on SOL, ETH
- Expected: SOL Sharpe >70% of BTC Sharpe

#### 12.3 Regime-Aware Validation
- Test in low/med/high volatility regimes separately
- Strategy should work (Sharpe >0.5) in all regimes

#### 12.4 Transaction Cost Sensitivity
- Test with 0.5x, 1x, 1.5x, 2x, 3x costs
- Requirement: Sharpe >1.0 with 2x costs

**Key Files:**
- `src/validation/walk_forward_validator.py`
- `src/validation/cross_symbol_validator.py`
- `src/validation/regime_validator.py`
- `src/validation/cost_sensitivity.py`

**→ Create:** `docs/PHASE_12_VALIDATION.md`

---

### PHASE 13: Production Readiness (3-4 Days)
**Prerequisites:** Phase 12 complete, all validation passed
**Objective:** Monitoring, logging, alerting for live trading

**Components:**

#### 13.1 Data Quality Monitoring
- Real-time gap detection
- Latency monitoring (<100ms)
- Continuous aggregate lag alerts

#### 13.2 Strategy Monitoring
- Live Sharpe calculation
- Drawdown tracking
- Fill rate vs expected
- Slippage actual vs estimate

#### 13.3 Model Degradation Detection
- Feature drift detection
- Prediction accuracy tracking
- Automatic model retraining triggers

#### 13.4 Execution Simulation
- Limit order fill probability modeling
- Queue position simulation
- Partial fill handling

**Key Files:**
- `src/monitoring/data_quality_monitor.py`
- `src/monitoring/strategy_monitor.py`
- `src/monitoring/model_monitor.py`
- `src/execution/order_simulator.py`

**Alerts:**
- Email/Slack on data gaps >10 min
- Kill switch on drawdown >20%
- Model retraining if Sharpe drops <0.5 for 7 days

**→ Create:** `docs/PHASE_13_PRODUCTION_READINESS.md`

---

### PHASE 14: Paper Trading (14-21 Days)
**Prerequisites:** Phase 13 complete
**Objective:** Validate in live markets WITHOUT real money

**Setup:**
1. Connect to exchange REST API (read-only)
2. Simulate order execution (no actual orders)
3. Track fills based on order book state
4. Log all decisions and compare to backtest

**Metrics to Track:**
- Live Sharpe vs backtest Sharpe
- Slippage actual vs estimated
- Fill rate vs expected
- Feature distributions vs training

**Success Criteria:**
- 30 consecutive days profitable
- Sharpe >1.0 (after costs)
- Slippage <2x backtest estimate
- No data quality issues

**Daily Review:**
```bash
python scripts/paper_trading_report.py --date $(date +%Y-%m-%d)

# Review:
# - PnL vs expected
# - Feature drift
# - Execution quality
# - Any anomalies
```

**GO/NO-GO for Live:**
- If paper trading fails → Debug and re-paper trade
- If succeeds → Proceed to Phase 15 with TINY size

**→ Create:** `docs/PHASE_14_PAPER_TRADING.md`

---

### PHASE 15: Live Deployment (Ongoing)
**Prerequisites:** Phase 14 passed (30+ days profitable)
**Objective:** Deploy to live markets with extreme caution

**Deployment Stages:**

#### Stage 1: Micro Size (Week 1-2)
- **Size:** 1% of backtest position size
- **Objective:** Validate execution in prod
- **Monitoring:** Hourly checks

#### Stage 2: Small Size (Week 3-4)
- **Size:** 5% of backtest size
- **Objective:** Build confidence
- **Monitoring:** Daily checks

#### Stage 3: Gradual Ramp (Month 2-3)
- **Size:** Increase 5% per week if profitable
- **Cap:** 50% of backtest size (safety buffer)

**Risk Management:**
- **Hard stop:** Max drawdown 20%
- **Position limits:** Max 50% in single asset
- **Correlation limits:** Max 70% in correlated positions
- **Volatility scaling:** Reduce size in high vol (>75th percentile)

**Monitoring Dashboard:**
```bash
# Real-time dashboard
streamlit run src/monitoring/live_dashboard.py

# Displays:
# - Current positions
# - PnL (daily, weekly, monthly)
# - Sharpe ratio (rolling 30-day)
# - Drawdown
# - Feature drift alerts
# - Execution quality
```

**Kill Switches:**
1. **Data quality:** Gap >10 min → Flatten all positions
2. **Drawdown:** >20% → Stop trading for 24h
3. **Model drift:** Feature distributions >3σ from training → Pause
4. **Execution:** Slippage >3x estimate → Reduce size 50%

**Weekly Review Checklist:**
- [ ] Review all trades (why entered, why exited)
- [ ] Check slippage vs estimate
- [ ] Analyze losing trades (was signal wrong or execution?)
- [ ] Feature drift analysis
- [ ] Model retraining if needed (monthly)

**→ Create:** `docs/PHASE_15_LIVE_DEPLOYMENT.md`

---

## 🎯 CRITICAL SUCCESS FACTORS

### 1. Features > Models
**70% of performance comes from feature engineering.**

- Spend 7-9 days on Phase 6 (features)
- Test each feature individually before combining
- Focus on mean reversion-specific features (half-life, fair value deviations)

### 2. Transaction Costs Will Kill You
**Assume 50-80% of gross returns disappear to costs.**

- Model costs dynamically (Phase 11)
- Test with 2x assumed costs
- Favor lower-frequency strategies
- Use passive execution when possible (earn spread)

### 3. Overfitting is Guaranteed Without Discipline
**Crypto has high noise-to-signal ratio.**

Defense:
- Cross-symbol validation (train BTC, test SOL)
- Regime-aware validation
- Feature selection (<50 features)
- Transaction cost sensitivity testing

### 4. Regime Awareness is Non-Negotiable
**Mean reversion FAILS in strong trends.**

- Build regime detector (ADX, volatility percentile)
- Kill switch during trending markets
- Test strategy in all regimes separately

### 5. Paper Trade for 30+ Days
**Backtest ≠ Live performance.**

- Validate fill assumptions
- Check slippage estimates
- Monitor feature drift
- Build confidence before risking capital

---

## 🚦 GO/NO-GO DECISION GATES

### Gate 0: POC (After Phase 0)
**Criteria:**
- [ ] Microstructure features improve Sharpe by +0.5
- [ ] Strategy profitable with 2x costs
- [ ] Top features have intuitive SHAP values

**Decision:** If ANY criterion fails → Do not proceed with full migration

### Gate 1: Backtest (After Phase 12)
**Criteria:**
- [ ] Out-of-sample Sharpe >1.5 (realistic costs)
- [ ] Sharpe >1.0 with 2x costs
- [ ] Cross-symbol validation passed (SOL Sharpe >70% of BTC)
- [ ] Works in all volatility regimes (Sharpe >0.5)

**Decision:** If ANY criterion fails → Iterate on features/strategies

### Gate 2: Paper Trading (After Phase 14)
**Criteria:**
- [ ] 30+ consecutive days profitable
- [ ] Live Sharpe >1.0
- [ ] Slippage <2x backtest estimate
- [ ] No critical data quality issues

**Decision:** If ANY criterion fails → Debug and re-paper trade

### Gate 3: Live Micro Size (After 2 weeks Phase 15 Stage 1)
**Criteria:**
- [ ] Profitable with 1% size
- [ ] Execution quality matches paper trading
- [ ] No unexpected behaviors

**Decision:** If passes → Ramp to 5% size. If fails → Back to paper trading.

---

## 📊 EXPECTED TIMELINE

### Optimistic (All Goes Well)
- **POC:** 1 week
- **Phases 1-13:** 6 weeks
- **Paper Trading:** 3 weeks (minimum)
- **Total:** ~10 weeks to live deployment

### Realistic (Some Iteration Required)
- **POC:** 1 week + 3 days iteration
- **Phases 1-13:** 7-8 weeks
- **Paper Trading:** 4-5 weeks (with debugging)
- **Total:** ~13-15 weeks to live deployment

### Pessimistic (Major Issues)
- **POC:** 2 weeks (multiple iterations)
- **Phases 1-13:** 10 weeks
- **Paper Trading:** 6-8 weeks (significant debugging)
- **Total:** ~18-20 weeks

**Most Likely:** 12-16 weeks from start to live trading with micro size

---

## 🔧 TROUBLESHOOTING GUIDE

### Common Issues & Solutions

#### POC Fails (Sharpe <1.0)
**Causes:**
- Features not predictive
- Overfitting to noise
- Transaction costs underestimated

**Solutions:**
1. Focus on strongest individual features (VPIN, price zscore)
2. Increase signal threshold (be more selective)
3. Longer holding periods (reduce turnover)
4. Try passive execution (earn spread)

#### Model Overfits (Great in-sample, poor out-of-sample)
**Causes:**
- Too many features
- Too deep trees
- Not enough regularization

**Solutions:**
1. Feature selection (reduce to top 30)
2. Reduce max_depth (try 4 instead of 6)
3. Increase lambda_l1/l2
4. Cross-symbol validation

#### Paper Trading Fails (Doesn't Match Backtest)
**Causes:**
- Fill assumptions wrong
- Slippage underestimated
- Feature drift
- Data quality issues

**Solutions:**
1. Log every decision and execution
2. Compare fill rates to backtest assumptions
3. Check feature distributions vs training
4. Validate data quality (gaps, latency)
5. Adjust cost model based on actuals

#### Live Trading Underperforms
**Causes:**
- Market regime changed
- Slippage worse than expected
- Competition (strategy decay)

**Solutions:**
1. Check current regime vs training
2. Reduce position size
3. Retrain model on recent data
4. Consider pausing strategy

---

## 📝 DAILY WORKFLOW (During Execution)

### For Claude Code:

**Start of Day:**
```bash
# 1. Check what phase we're on
cat EXECUTION_GUIDE.md | grep "PHASE.*in progress"

# 2. Review phase objectives
cat docs/PHASE_XX_NAME.md | head -50

# 3. Check todos
cat TODO.md  # Or review previous session notes
```

**During Execution:**
```bash
# 4. Implement tasks from phase guide
# Follow step-by-step instructions in phase document

# 5. Run verification commands
# Each phase has specific verification steps

# 6. Commit frequently
git add .
git commit -m "[PHASE X] Task description"
```

**End of Day:**
```bash
# 7. Update progress
echo "Completed: Task X, Y, Z" >> PROGRESS.md

# 8. Note any blockers
echo "Blockers: ..." >> PROGRESS.md

# 9. Plan next session
echo "Next: Task A, B" >> PROGRESS.md
```

---

## 🎓 LEARNING RESOURCES

### Order Book Microstructure
- Cont, Stoikov, Talreja (2010): "A Stochastic Model for Order Book Dynamics"
- Easley, López de Prado, O'Hara (2012): "VPIN and Flash Crashes"

### Transaction Costs
- Almgren & Chriss (2000): "Optimal Execution of Portfolio Transactions"
- Kyle (1985): "Continuous Auctions and Insider Trading"

### Mean Reversion
- Ornstein-Uhlenbeck process for half-life estimation
- Hurst exponent for mean reversion detection

### Overfitting Prevention
- López de Prado (2018): "Advances in Financial Machine Learning"
- Purged k-fold cross-validation for time series

---

## ✅ FINAL CHECKLIST (Before Live Trading)

### Technical
- [ ] All 15 phases completed
- [ ] All tests passing (unit + integration)
- [ ] Documentation up to date
- [ ] Monitoring dashboard working
- [ ] Kill switches tested

### Validation
- [ ] Out-of-sample Sharpe >1.5
- [ ] Cross-symbol validation passed
- [ ] Regime-aware validation passed
- [ ] Transaction cost sensitivity passed (Sharpe >1.0 with 2x)

### Paper Trading
- [ ] 30+ days profitable
- [ ] Sharpe >1.0 (after costs)
- [ ] Slippage <2x estimate
- [ ] Fill rates match expectations
- [ ] No data quality issues

### Risk Management
- [ ] Position limits configured
- [ ] Drawdown limits set
- [ ] Correlation limits set
- [ ] Kill switches active
- [ ] Monitoring alerts configured

### Operational
- [ ] Exchange API keys secured
- [ ] Database backups configured
- [ ] Model versioning system
- [ ] Runbook for common issues
- [ ] On-call schedule (if team)

---

## 🚀 QUICK START

**For Immediate Execution:**

1. **Read POC Implementation** (Days 1-7)
   ```bash
   cat MIGRATION_EXECUTION_PLAN.md | head -1200
   ```

2. **Start Phase 0, Day 1** (TimescaleDB POC)
   ```bash
   # Follow instructions exactly
   # Create docker-compose-poc.yml
   # Run benchmark
   ```

3. **Execute Day by Day**
   - Don't skip ahead
   - Verify each step
   - Commit frequently

4. **Make GO/NO-GO Decision on Day 7**
   - Review all success criteria
   - If GO → Proceed to Phase 1
   - If NO-GO → Iterate or pivot

---

## 📞 SUPPORT & FEEDBACK

**During Execution:**
- Document issues in `ISSUES.md`
- Note learnings in `LESSONS_LEARNED.md`
- Update this guide with improvements

**After Completion:**
- Conduct retrospective
- Document what worked / didn't work
- Share learnings with community

---

**Last Updated:** 2025-11-30
**Version:** 3.0
**Status:** ✅ READY FOR EXECUTION

---

END OF EXECUTION GUIDE
