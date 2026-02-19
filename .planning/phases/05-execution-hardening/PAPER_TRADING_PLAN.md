# HYDRA Paper Trading Validation Plan

## 1. Overview

This document defines the 4-week minimum paper trading validation period required before any live capital is deployed (EXEC-05). The purpose of this period is to:

- **Validate the execution pipeline** end-to-end: BrokerGateway -> RiskGate -> OrderManager -> FillJournal
- **Calibrate the slippage model** by comparing predicted slippage (from `estimate_slippage()`) against actual IB paper-trading fills
- **Prove the self-healing cycle works**: the agent loop must detect drift, generate hypotheses, run experiments, and either promote or roll back a candidate model -- all without human intervention

No live capital will be deployed until ALL gate conditions in Section 5 are met and a human review confirms readiness.

## 2. Monitoring Schedule

### Daily Checks

- [ ] Fill journal has new entries: `hydra fill-report --last 24h`
- [ ] Review any risk gate blocks in logs (circuit breaker triggers)
- [ ] Verify agent loop ran successfully (check `~/.hydra/agent_state.json`)
- [ ] Confirm broker connection was maintained (no extended disconnects)
- [ ] Check for any unhandled exceptions in `structlog` output

### Weekly Checks

- [ ] Run slippage reconciliation: `hydra fill-report --reconcile`
  - Track bias trend (should converge toward 0)
  - Track correlation trend (should increase toward 1.0)
  - Track RMSE trend (should decrease or stabilize)
- [ ] Review agent cycle logs for drift detection events and healing actions
  - Were any hypotheses generated?
  - Were any experiments run?
  - Were any candidates promoted or rolled back?
- [ ] Check circuit breaker trigger history
  - Were any breakers triggered? Were they appropriate?
  - Did cooldown periods work correctly?
- [ ] Verify reconnection handling
  - Were there any disconnects? Did auto-reconnect succeed?

### End-of-Week Summary

Record the following metrics at the end of each week:

| Metric | Week 1 | Week 2 | Week 3 | Week 4 |
|--------|--------|--------|--------|--------|
| Paper P&L (cumulative) | | | | |
| Number of trades executed | | | | |
| Number of risk gate blocks | | | | |
| Slippage bias (actual - predicted) | | | | |
| Slippage RMSE | | | | |
| Slippage correlation | | | | |
| Agent drift detections | | | | |
| Agent experiments run | | | | |
| Agent promotions/rollbacks | | | | |
| Disconnects / reconnects | | | | |
| Unhandled errors | | | | |

## 3. Success Metrics (Gate Conditions for Live Capital)

All of the following must be achieved before moving to live trading:

1. **Continuous operation**: Minimum 4 weeks of paper trading without manual intervention required for the trading pipeline to function
2. **Slippage model calibrated**: `reconciler.is_model_calibrated()` returns `True`
   - Absolute bias < 0.5
   - Pearson correlation > 0.3
3. **Self-healing cycle proven**: At least 1 complete self-healing cycle observed
   - Drift detected (KS test p < 0.05 or prediction accuracy drop > 10%)
   - Hypothesis generated (mutation playbook or LLM-assisted)
   - Experiment executed (candidate trained and evaluated)
   - Promotion or rollback decision made (both are valid outcomes)
4. **No unrecoverable errors**: Zero crashes, data corruption, or zombie connections
5. **Circuit breakers verified**: Correct triggering observed when thresholds are approached during paper fills
6. **Reconnection handling verified**: At least 1 successful auto-reconnect observed or simulated

## 4. Self-Healing Criteria

### What Constitutes Drift

- **Distribution shift**: KS test p-value < 0.05 on recent returns vs historical baseline
- **Prediction accuracy drop**: Accuracy drops > 10% from rolling baseline (measured via agent observe phase)
- **Regime change**: CUSUM detector triggers on streaming performance metrics

### Expected Healing Cadence

- Agent loop runs daily (triggered by APScheduler at 2 PM CT)
- Drift checks occur every cycle; actual drift detection expected ~weekly depending on market regime
- In quiet markets, the agent may go several days without detecting actionable drift (this is normal)

### What Counts as a Successful Healing Cycle

The cycle is successful if **either** of these outcomes occurs:

1. **Promotion**: Candidate model outperforms champion across 3-of-5 PromotionEvaluator windows, is promoted to champion, and the system continues operating normally
2. **Rollback**: Candidate model fails to outperform, is correctly rolled back, and the champion continues operating (this proves the safety net works)

Both outcomes validate the self-healing pipeline. The key is that the cycle **completes** without human intervention.

## 5. Live Capital Gate Conditions

ALL of the following must be satisfied before live trading begins:

- [ ] 4+ weeks of stable paper trading completed
- [ ] Slippage model calibrated (`is_model_calibrated() == True`)
- [ ] At least 1 self-healing cycle completed (promote or rollback)
- [ ] Human review of all metrics in the weekly summary table (Section 2)
- [ ] Human review of fill journal data quality
- [ ] `HYDRA_LIVE_CONFIRMED=true` environment variable set explicitly
- [ ] IB Gateway configured on live port (4001) with appropriate permissions
- [ ] Position sizing verified as conservative (single-contract initial)

This is a **blocking gate**. No automated process can bypass it. The `HYDRA_LIVE_CONFIRMED` env var must be set manually after human review.

## 6. Escalation Procedures

### Paper Trading Fails to Connect (> 24 hours)

1. Check IB Gateway is running: `ps aux | grep -i ibgateway`
2. Verify port 4002 is listening: `lsof -i :4002`
3. Restart IB Gateway (or IBC if using headless operation)
4. Verify API settings: Socket API enabled, port 4002, localhost allowed
5. Check IB account status (maintenance, forced logout, etc.)
6. If using IBC: check `~/ibc/logs/` for error details

### Slippage Model Never Calibrates

1. Check fill count: `hydra fill-report --count` (need >= 10 fills minimum)
2. Inspect fill data quality: look for outlier fills, zero-slippage entries, or data corruption
3. If fills are sparse: increase trading frequency or extend paper period
4. If bias is consistently large: investigate whether `estimate_slippage()` parameters (impact_coefficient, spread) match actual market conditions
5. If correlation is low: the market impact model may need different parameterization for this specific contract

### Self-Healing Never Triggers

1. This is expected in stable market regimes -- the absence of drift is not a failure
2. To verify the pipeline works, artificially inject a drift scenario:
   - Manually edit recent prediction accuracy to simulate a > 10% drop
   - Or inject synthetic returns with a distributional shift
3. Observe that the agent detects the injected drift and runs through the full cycle
4. After verification, extend the paper period if no natural drift occurs
5. Document the artificial test as part of the validation evidence

### Unrecoverable Error Occurs

1. Capture full stack trace from structlog output
2. Check SQLite journal integrity: `sqlite3 fills.db "PRAGMA integrity_check"`
3. If data corruption: restore from last known-good backup (WAL mode should prevent most corruption)
4. Fix the root cause before resuming paper trading
5. Reset the 4-week clock -- the continuous operation requirement restarts

## 7. Starting Paper Trading

To begin the 4-week paper trading period:

```bash
# 1. Ensure IB Gateway is running on port 4002 (paper)
# 2. Verify connectivity
hydra paper-trade --dry-run

# 3. Start the paper trading runner
# (long-running process -- use screen/tmux or systemd)
python -m hydra.execution.runner

# 4. Monitor logs
tail -f ~/.hydra/logs/runner.log
```

The runner will execute one trading cycle per day at 2:00 PM Central Time (after the 1:15 PM CT pit close for agricultural futures).

## 8. Transitioning to Live

After all gate conditions are met:

```bash
# 1. Update IB Gateway to use live port 4001
# 2. Set the confirmation env var
export HYDRA_LIVE_CONFIRMED=true

# 3. Start with conservative sizing
# (single contract, same daily schedule)
python -m hydra.execution.runner --mode live --yes-i-mean-live
```

Initial live trading uses single-contract sizing. Position sizing can be increased only after additional live validation confirms the system performs as expected under real market conditions.
