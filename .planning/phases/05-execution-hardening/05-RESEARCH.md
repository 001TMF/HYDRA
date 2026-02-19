# Phase 5: Execution + Hardening - Research

**Researched:** 2026-02-19
**Domain:** Interactive Brokers API integration, order management, risk middleware, paper trading validation
**Confidence:** MEDIUM-HIGH

## Summary

Phase 5 connects the fully autonomous HYDRA agent loop to Interactive Brokers for paper trading, then validates that simulated performance matches real fills over 4+ weeks. The primary integration library is `ib_async` 2.1.0, the actively maintained successor to `ib_insync` (whose original creator passed away in early 2024). The critical research flag from the roadmap -- ib_insync maintenance status -- is now resolved: **use `ib_async`**, not `ib_insync`.

The architecture follows a layered pattern: a broker abstraction layer wraps `ib_async`, an order manager handles smart routing (limit orders with patience, custom TWAP slicing for larger positions), and risk management sits as mandatory middleware in the execution pipeline -- every order passes through circuit breakers and position sizing checks before reaching the broker. Fill logging captures timestamps, actual slippage, and execution details to a SQLite-backed fill journal for post-hoc validation against the existing simulated slippage model.

IB's paper trading environment has known limitations: fills simulate from top-of-book only (no deep book), VWAP orders are unsupported, and stops are always simulated. These are acceptable for HYDRA's use case (thin commodity futures with limit orders), but the slippage validation methodology must account for the fact that paper fills are optimistic compared to live. The 4-week stability period with self-healing cycle validation is the final gate before any live capital.

**Primary recommendation:** Use `ib_async` 2.1.0 with IB Gateway + IBC for headless operation. Build a broker abstraction layer so the execution path is identical for paper and live trading. Implement custom TWAP slicing rather than relying on IB's built-in TWAP algo (which may have limitations for thin commodity futures). Risk management as middleware is non-negotiable -- enforce at the architecture level with a `RiskGate` class that sits between the order manager and broker layer.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EXEC-01 | Paper trading pipeline uses IB paper account with same execution path as live | `ib_async` 2.1.0 connects to IB Gateway paper account (port 4002) or TWS paper (port 7497). Broker abstraction layer ensures identical code path for paper and live -- only the connection port differs. IBC enables headless Gateway operation. |
| EXEC-02 | All paper trading fills logged with timestamps and actual slippage for model validation | `ib_async` Trade objects provide fill price, timestamp, and execution details. Build a FillJournal (SQLite) that logs every fill alongside the model's predicted slippage from `estimate_slippage()`. Post-hoc analysis compares actual vs predicted slippage distributions. |
| EXEC-03 | Order management implements smart order routing (limit orders with patience, TWAP for larger positions) | For thin commodity futures: (1) default to limit orders placed at mid-price with configurable patience timeout, (2) for larger orders exceeding volume threshold, implement custom TWAP slicing with randomized intervals. IB's native TWAP is available for US futures but custom implementation gives more control for thin markets. |
| EXEC-04 | Risk management runs as middleware in execution path (not optional check) | `RiskGate` class wraps the broker layer. Every `submit_order()` call passes through `RiskGate.check()` which runs the existing `CircuitBreakerManager.check_trade()` and `volume_capped_position()` checks. If any check fails, the order is rejected before reaching the broker. This is enforced architecturally -- there is no code path that bypasses risk checks. |
| EXEC-05 | 4+ weeks of stable paper trading with at least one successful self-healing cycle before any live capital | The paper trading runner invokes `AgentLoop.run_cycle()` on a schedule (daily after market close for EOD strategy). Stability metrics (equity drawdown, Sharpe window, fill rate) are monitored. Self-healing cycle requires the agent to detect drift, diagnose, propose mutation, test in sandbox, and promote -- all while paper trading continues. Success = promoted model holds for 30+ days. |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ib_async | 2.1.0 | IB API wrapper (sync/async) | Actively maintained successor to ib_insync; 1.4k stars; built-in reconnection, order lifecycle tracking, asyncio-native. BSD license. Installed via `pip install ib_async`. |
| IBC | 3.23.0 | Headless IB Gateway automation | Automates login, 2FA mobile auth, dialog handling, daily restarts. Required for unattended operation. 39 releases, actively maintained. |
| IB Gateway | latest | Headless IB connection endpoint | Lighter than TWS, no GUI overhead, designed for automated systems. Paper account on port 4002, live on port 4001. |

### Supporting (already in project)

| Library | Version | Purpose | How Used in Phase 5 |
|---------|---------|---------|---------------------|
| structlog | >=24.1 | Structured logging | Fill logging, order lifecycle events, risk gate decisions |
| apscheduler | >=3.10 | Job scheduling | Schedule daily agent cycles, periodic health checks |
| sqlite3 | stdlib | Fill journal storage | FillJournal persistence (same pattern as ExperimentJournal) |
| numpy | >=1.26 | Slippage analysis | Compare actual vs predicted slippage distributions |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ib_async 2.1.0 | IB's new Sync Wrapper (TWS API 10.40) | IB's official sync wrapper is still in beta, limited documentation, no community ecosystem. ib_async is production-tested with 7+ years of community usage (counting ib_insync history). |
| ib_async 2.1.0 | ib_insync 0.9.86 (original) | ib_insync is unmaintained since creator's passing (early 2024). Still works but won't receive bug fixes. ib_async is API-compatible drop-in replacement. |
| ib_async 2.1.0 | Raw ibapi (IB's native Python package) | ibapi is callback-based, verbose, error-prone. ib_async wraps it into clean sync/async interface. No reason to use raw ibapi directly. |
| Custom TWAP | IB's built-in TWAP algo | IB TWAP is available for US futures, but thin commodity markets need finer control over slice sizing relative to volume. Custom implementation allows volume-adaptive slicing with randomized intervals specific to HYDRA's volume constraints. |

**Installation:**
```bash
uv add ib_async
# IBC installed separately (not a pip package) -- download from GitHub releases
```

## Architecture Patterns

### Recommended Project Structure

```
src/hydra/
├── execution/               # NEW: Phase 5 execution layer
│   ├── __init__.py
│   ├── broker.py            # BrokerGateway: ib_async wrapper with reconnection
│   ├── order_manager.py     # OrderManager: limit patience + TWAP slicing
│   ├── risk_gate.py         # RiskGate: mandatory pre-trade middleware
│   ├── fill_journal.py      # FillJournal: SQLite fill logging + slippage tracking
│   ├── reconciler.py        # SlippageReconciler: actual vs predicted comparison
│   └── runner.py            # PaperTradingRunner: orchestrates daily cycle
├── risk/                    # EXISTING: extended in Phase 5
│   ├── circuit_breakers.py  # CircuitBreakerManager (already exists)
│   ├── position_sizing.py   # fractional_kelly, volume_capped_position (exists)
│   └── slippage.py          # estimate_slippage (exists, used for comparison)
├── agent/
│   └── loop.py              # AgentLoop (exists, invoked by runner)
└── cli/
    └── app.py               # Extended with paper-trading commands
```

### Pattern 1: Broker Abstraction Layer

**What:** A `BrokerGateway` class that wraps `ib_async.IB` and provides a clean interface for the rest of HYDRA. Handles connection management, reconnection, and translates between HYDRA's internal types and IB's contract/order types.

**When to use:** Always -- all IB interaction goes through this single class.

**Example:**
```python
# Source: ib_async 2.1.0 docs + HYDRA conventions
from ib_async import IB, Future, LimitOrder, Trade

class BrokerGateway:
    """Wraps ib_async.IB with reconnection, health checks, and type translation."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 1):
        self.ib = IB()
        self._host = host
        self._port = port  # 4002 = Gateway paper, 4001 = Gateway live
        self._client_id = client_id

    async def connect(self) -> None:
        await self.ib.connectAsync(self._host, self._port, clientId=self._client_id)
        self.ib.disconnectedEvent += self._on_disconnect

    async def _on_disconnect(self) -> None:
        """Auto-reconnect on disconnect."""
        while not self.ib.isConnected():
            try:
                await self.ib.connectAsync(
                    self._host, self._port, clientId=self._client_id
                )
            except Exception:
                await asyncio.sleep(5)

    async def submit_order(self, contract: Future, order: LimitOrder) -> Trade:
        """Submit order and return Trade object for monitoring."""
        trade = self.ib.placeOrder(contract, order)
        return trade

    async def get_fills(self) -> list:
        """Return all fills from current session."""
        return self.ib.fills()
```

### Pattern 2: Risk Gate as Mandatory Middleware

**What:** A `RiskGate` class that wraps the broker and enforces all risk checks before any order reaches IB. There is no code path that bypasses it.

**When to use:** Every order submission. The `OrderManager` calls `RiskGate.submit()`, which checks circuit breakers, position limits, and volume caps before delegating to `BrokerGateway.submit_order()`.

**Example:**
```python
class RiskGate:
    """Mandatory pre-trade risk check middleware (EXEC-04).

    Every order passes through here. No bypass path exists.
    """

    def __init__(self, broker: BrokerGateway, breakers: CircuitBreakerManager):
        self._broker = broker
        self._breakers = breakers

    async def submit(
        self,
        contract,
        order,
        daily_pnl: float,
        peak_equity: float,
        current_equity: float,
        position_value: float,
        trade_loss: float,
    ) -> Trade | None:
        """Check risk then submit. Returns None if blocked."""
        allowed, triggered = self._breakers.check_trade(
            daily_pnl, peak_equity, current_equity, position_value, trade_loss,
        )
        if not allowed:
            logger.warning("order_blocked_by_risk", triggered=triggered)
            return None
        return await self._broker.submit_order(contract, order)
```

### Pattern 3: Smart Order Routing (Limit with Patience + TWAP)

**What:** For thin commodity futures, the default order type is a limit order placed at mid-price with a configurable patience window. If the order doesn't fill within the patience period, it steps the price toward the market. For larger orders (exceeding a volume threshold), the system slices into multiple smaller limit orders spread over time (custom TWAP).

**When to use:** All order submissions go through the `OrderManager` which decides the routing strategy based on order size relative to average daily volume.

**Example:**
```python
class OrderManager:
    """Smart order routing for thin futures markets (EXEC-03)."""

    PATIENCE_SECONDS = 300       # 5 min patience for limit orders
    TWAP_VOLUME_THRESHOLD = 0.01 # Orders > 1% of ADV get TWAP treatment
    TWAP_SLICES = 5              # Number of time slices
    TWAP_JITTER_PCT = 0.20       # Randomize slice timing by +/- 20%

    async def route_order(self, symbol, direction, n_contracts, adv, mid_price):
        participation_rate = n_contracts / max(adv, 1)

        if participation_rate > self.TWAP_VOLUME_THRESHOLD:
            return await self._twap_slice(symbol, direction, n_contracts, mid_price)
        else:
            return await self._limit_with_patience(symbol, direction, n_contracts, mid_price)

    async def _limit_with_patience(self, symbol, direction, n_contracts, mid_price):
        """Place limit at mid, wait PATIENCE_SECONDS, then step toward market."""
        # Place initial limit order at mid-price
        # If not filled after patience window, modify to best bid/ask
        # If still not filled, modify to cross spread
        ...

    async def _twap_slice(self, symbol, direction, n_contracts, mid_price):
        """Slice order into N time-spaced limit orders with jitter."""
        slice_size = n_contracts // self.TWAP_SLICES
        remainder = n_contracts % self.TWAP_SLICES
        # Each slice: limit order at current mid, patience timeout
        # Randomize intervals by TWAP_JITTER_PCT to reduce predictability
        ...
```

### Pattern 4: Fill Journal for Slippage Validation

**What:** A SQLite-backed journal that logs every fill with its predicted slippage (from `estimate_slippage()`) and actual slippage (from the IB fill price vs mid at order entry). Enables post-hoc comparison.

**When to use:** Every fill event triggers a journal entry.

**Example:**
```python
@dataclass
class FillRecord:
    timestamp: str           # ISO 8601
    symbol: str
    direction: int           # +1 long, -1 short
    n_contracts: int
    order_price: float       # mid-price at order entry
    fill_price: float        # actual fill price from IB
    predicted_slippage: float  # from estimate_slippage()
    actual_slippage: float   # abs(fill_price - order_price) per contract
    volume_at_fill: float    # market volume at time of fill
    spread_at_fill: float    # bid-ask spread at time of fill
    fill_latency_ms: float   # time from order to fill
```

### Anti-Patterns to Avoid

- **Direct IB calls from agent loop:** The agent loop must never call `ib_async` directly. All execution goes through `OrderManager -> RiskGate -> BrokerGateway`. This ensures risk checks are always enforced and the broker layer can be swapped (e.g., for backtesting).
- **Optional risk checks:** Risk management must not be a flag or optional call. The `RiskGate` is structurally mandatory -- there is no `submit_order()` method that bypasses it.
- **Market orders in thin markets:** Never use market orders for thin commodity futures. The spread and slippage can be catastrophic. Always use limit orders with patience.
- **Relying on IB paper fill realism:** Paper fills are top-of-book simulated. Do not treat paper trading performance as accurate prediction of live performance. The slippage reconciler exists specifically to quantify the gap.
- **Single-threaded blocking on IB:** Use `ib_async`'s asyncio support. The IB connection uses a network event loop that must not be blocked by model training or other CPU-bound work.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| IB API connection management | Custom socket handler | `ib_async.IB` with connectAsync | ib_async handles the binary protocol, heartbeats, message parsing, type conversion. 7+ years of community hardening. |
| Order lifecycle tracking | Custom state machine for order status | `ib_async.Trade` objects | Trade objects auto-update with fills, cancels, status changes. Event-driven callbacks built in. |
| IB contract resolution | Manual contract specification | `ib_async.IB.qualifyContracts()` | Resolves ambiguous contract specs to fully qualified IB contracts. Handles exchange routing. |
| Fill event streaming | Polling loop for execution reports | `ib_async` orderStatusEvent / execDetailsEvent | Event-driven callbacks fire on every fill. No polling needed. |
| Headless IB Gateway operation | Custom login automation | IBC 3.23.0 | Handles login dialogs, 2FA mobile, daily restarts, crash recovery. Battle-tested. |
| Slippage model math | New slippage estimator | Existing `hydra.risk.slippage.estimate_slippage()` | Already implements Almgren & Chriss square-root impact model. Phase 5 validates it, doesn't replace it. |
| Circuit breaker logic | New risk check system | Existing `hydra.risk.circuit_breakers.CircuitBreakerManager` | Already has 4 independent breakers with state machine. Phase 5 wraps it in RiskGate middleware. |
| Position sizing | New sizing algorithm | Existing `hydra.risk.position_sizing` (Kelly + volume cap) | Already implements fractional Kelly with volume cap. Phase 5 feeds it live data. |

**Key insight:** Phase 5 is primarily an **integration** phase. The risk management, position sizing, slippage model, and agent loop already exist. Phase 5 wraps them in an execution layer that connects to IB and validates their accuracy against real fills. New code should be minimal -- mostly broker abstraction, order routing, fill logging, and the paper trading runner orchestrator.

## Common Pitfalls

### Pitfall 1: IB Gateway Disconnects During Trading Hours

**What goes wrong:** IB Gateway disconnects at ~11:45 PM ET daily for server reset. If your system doesn't handle reconnection, it silently stops trading.
**Why it happens:** IB performs nightly resets. The disconnect is expected but can cause issues if not handled.
**How to avoid:** Use IBC for automatic restart management. Implement reconnection logic in `BrokerGateway` with exponential backoff. Use `ib_async`'s `disconnectedEvent` callback.
**Warning signs:** Orders stuck in "PendingSubmit" state, no fill events received, `isConnected()` returns False.

### Pitfall 2: Client ID Conflicts

**What goes wrong:** Two processes connect to IB Gateway with the same client ID. The second connection kicks out the first, causing the first to lose all order tracking state.
**Why it happens:** Each IB API connection needs a unique client ID. If you run paper trading and a diagnostic tool simultaneously, they'll conflict.
**How to avoid:** Assign fixed client IDs: 1 for paper trading runner, 2 for diagnostic tools, 3 for CLI ad-hoc queries. Document the allocation.
**Warning signs:** "Already connected" errors, sudden disconnection of the trading process, duplicate order submissions.

### Pitfall 3: Paper Trading Fill Optimism

**What goes wrong:** Paper trading shows better performance than live trading would achieve, because paper fills simulate from top-of-book without realistic queue priority or market impact.
**Why it happens:** IB's paper trading engine assumes your order gets filled at the displayed price without moving the market. In thin commodity futures, your order IS the market.
**How to avoid:** Never trust paper trading P&L as predictive of live performance. Use the `SlippageReconciler` to compare predicted vs actual slippage. Build a pessimistic adjustment factor from the paper trading period. Start live with minimal size.
**Warning signs:** Paper Sharpe significantly exceeds backtest Sharpe; fill rates near 100% on limit orders.

### Pitfall 4: Blocking the asyncio Event Loop

**What goes wrong:** Model training, SHAP calculations, or heavy pandas operations block the asyncio event loop, causing IB heartbeat timeouts and disconnection.
**Why it happens:** `ib_async` runs on asyncio. CPU-bound work on the same thread blocks the event loop.
**How to avoid:** Run CPU-bound work (model training, diagnosis) in a separate process (the existing `ExperimentRunner` already uses subprocess isolation). Use `asyncio.to_thread()` for moderate blocking operations. Never call synchronous model training in the async trading loop.
**Warning signs:** IB connection drops during agent cycle, heartbeat timeout warnings in logs.

### Pitfall 5: Confusing Paper and Live Ports

**What goes wrong:** System accidentally connects to live trading port instead of paper trading port, executing real trades with real money.
**Why it happens:** Port 4001 (live) vs 4002 (paper) is a single digit difference. Configuration error during development.
**How to avoid:** Default configuration MUST be paper trading port (4002). Live port requires explicit `--live` flag AND a confirmation prompt in CLI. Configuration is validated at startup with a log message confirming paper/live mode. Environment variable `HYDRA_TRADING_MODE=paper|live` with default `paper`.
**Warning signs:** Account values don't match paper account, real P&L appearing in IB statements.

### Pitfall 6: Order State Desynchronization

**What goes wrong:** Local order state diverges from IB's server state after a reconnection, leading to duplicate orders or phantom positions.
**Why it happens:** ib_async v2.0.0 fixed "phantom orders" from cache deletion bugs, but reconnection after a long disconnect can still cause state gaps.
**How to avoid:** After reconnection, call `ib.reqOpenOrders()` and `ib.reqPositions()` to resynchronize. Reconcile local state with broker state before resuming trading. Log any discrepancies.
**Warning signs:** Position count doesn't match expected, orders reported as filled locally but open on broker.

## Code Examples

Verified patterns from official sources:

### Connecting to IB Gateway (Paper Account)

```python
# Source: ib_async 2.1.0 docs
from ib_async import IB, Future, LimitOrder

ib = IB()

# Paper trading via Gateway
await ib.connectAsync("127.0.0.1", 4002, clientId=1)

# Define a futures contract (e.g., oat futures on CBOT)
contract = Future(symbol="ZO", exchange="CBOT", lastTradeDateOrContractMonth="202603")
await ib.qualifyContractsAsync(contract)

# Place a limit buy order
order = LimitOrder("BUY", 1, 350.0)  # 1 contract at 350.0
trade = ib.placeOrder(contract, order)

# Monitor fill events
def on_fill(trade, fill):
    print(f"Filled: {fill.execution.shares} @ {fill.execution.price}")

trade.fillEvent += on_fill
```

### Monitoring Order Lifecycle

```python
# Source: ib_async 2.1.0 docs
def on_order_status(trade):
    """Track order status changes."""
    status = trade.orderStatus
    print(f"Order {trade.order.orderId}: {status.status} "
          f"filled={status.filled} remaining={status.remaining}")

trade.statusEvent += on_order_status

# After order completes, extract fill details
for fill in trade.fills:
    exec_detail = fill.execution
    print(f"Fill: {exec_detail.shares} @ {exec_detail.price} "
          f"time={exec_detail.time}")
```

### Getting Account Summary for Risk Checks

```python
# Source: ib_async 2.1.0 docs
# Get account values for risk gate
account_values = ib.accountSummary()
positions = ib.positions()

# Extract equity and daily P&L
net_liquidation = next(
    (v.value for v in account_values if v.tag == "NetLiquidation"), 0
)
daily_pnl = next(
    (v.value for v in account_values if v.tag == "DailyPnL"), 0
)
```

### IBC Configuration for Headless Gateway

```ini
# Source: IBC 3.23.0 docs (config.ini)
# Paper trading Gateway configuration
FIX=no
IbLoginId=your_username
IbPassword=your_password
TradingMode=paper
IbDir=/opt/ibc
AcceptIncomingConnectionAction=accept
ExistingSessionDetectedAction=primary
# Auto-restart daily at 11:45 PM ET (before IB nightly reset)
ClosedownAt=23:45
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ib_insync 0.9.86 | ib_async 2.1.0 | 2024-03 (fork), 2025-12 (2.1.0 release) | ib_insync unmaintained; ib_async is API-compatible successor with active development |
| Raw ibapi callbacks | ib_async sync/async wrapper | 2017 (original ib_insync) | Eliminates callback hell; provides clean pythonic API |
| IB Sync Wrapper (official) | Beta since TWS API 10.40 | 2025-09 | IB's official sync wrapper exists but is beta, limited docs, not yet production-recommended |
| Fixed slippage models | Volume-adaptive slippage + live validation | Ongoing best practice | HYDRA already has Almgren-Chriss model; Phase 5 validates against real fills |

**Deprecated/outdated:**
- **ib_insync (pip package):** Unmaintained since early 2024. Replace with `ib_async`. API is nearly identical -- rename imports from `ib_insync` to `ib_async`.
- **IBPy / ibpy2:** Ancient wrappers, completely superseded. Do not use.
- **IB TWS Sync Wrapper:** Still beta (as of Feb 2026). Not recommended for production yet.

## Open Questions

1. **IB TWAP algo availability for specific thin commodity futures (oats, lean hogs, ethanol)**
   - What we know: IB TWAP is listed as available for "US futures" generically. Oat futures (ZO) trade on CBOT/Globex which is US.
   - What's unclear: Whether IB TWAP works well for contracts with <500 daily volume. The algo may not slice effectively in extremely thin markets.
   - Recommendation: Implement custom TWAP as primary approach (gives volume-aware control). Test IB native TWAP as secondary option during paper trading. Compare fill quality.

2. **Paper trading fill quality for thin commodity futures specifically**
   - What we know: IB paper fills use top-of-book simulation. Fills are not representative of live execution in thin markets.
   - What's unclear: Exactly how bad the gap is for oat/lean hog futures specifically. No published data on paper vs live slippage for these specific markets.
   - Recommendation: Run paper trading for the full 4+ weeks, build empirical slippage data, then apply a pessimistic multiplier (1.5-2x) to predicted slippage when evaluating whether to go live.

3. **IB Gateway 2FA handling for fully unattended operation**
   - What we know: IBC 3.23.0 supports 2FA via IBKR Mobile. You must respond to the mobile prompt.
   - What's unclear: Whether IB enforces 2FA re-authentication on paper accounts at weekly intervals. If so, fully unattended 4+ week operation may require periodic human interaction.
   - Recommendation: Test during initial setup. If 2FA is required periodically, document the schedule and plan for it. Consider requesting paper-account-only API access if IB offers it.

4. **Daily schedule interaction between agent loop and market hours**
   - What we know: Thin commodity futures have limited trading hours (e.g., ZO Globex: 7pm-7:45am CT, pit: 9:30am-1:15pm CT). Agent loop runs daily.
   - What's unclear: Optimal timing for agent cycle -- run after market close? During overnight session? How to handle the gap between pit close and Globex open?
   - Recommendation: Run agent observe/diagnose after the pit session close (1:15 PM CT). Run order execution during Globex session for better fill probability. Configure APScheduler jobs accordingly.

## Sources

### Primary (HIGH confidence)
- [ib_async 2.1.0 PyPI](https://pypi.org/project/ib_async/) - Version 2.1.0, released 2025-12-08, Python 3.10+
- [ib_async GitHub](https://github.com/ib-api-reloaded/ib_async) - 1.4k stars, 857 commits, actively maintained by Matt Stancliff
- [ib_async changelog](https://ib-api-reloaded.github.io/ib_async/changelog.html) - Full version history v1.0.0 to v2.1.0
- [ib_async 2.1.0 documentation](https://ib-api-reloaded.github.io/ib_async/) - API reference, connection patterns, order management
- [IBC 3.23.0 GitHub](https://github.com/IbcAlpha/IBC) - Headless IB automation, released 2025-07-03
- [IB TWS API IB Algorithms](https://interactivebrokers.github.io/tws-api/ibalgos.html) - TWAP parameters, algo availability
- [IB Paper Trading vs Live](https://www.interactivebrokers.com/campus/trading-lessons/paper-trading-vs-live-trading-whats-the-difference/) - Fill simulation limitations

### Secondary (MEDIUM confidence)
- [IB TWAP Algo page](https://www.interactivebrokers.com/campus/trading-lessons/time-weighted-average-price-twap/) - Available for US equities, options, futures, forex
- [IB New Sync Wrapper](https://www.interactivebrokers.com/campus/ibkr-quant-news/the-new-synchronous-wrapper-for-tws-api/) - Beta status, TWS API 10.40+
- [Paper vs Live Slippage Analysis](https://markrbest.github.io/paper-vs-live/) - Methodology for comparing simulated vs actual slippage

### Tertiary (LOW confidence)
- IB TWAP availability for specific thin commodity futures (oats, lean hogs) - Not explicitly confirmed in any source. Generically listed as "US futures."
- IB paper account 2FA re-authentication frequency - Not documented. Needs empirical testing.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - ib_async 2.1.0 is well-documented, actively maintained, and the clear successor to ib_insync. IBC is battle-tested for headless operation. Both verified from primary sources.
- Architecture: MEDIUM-HIGH - The layered pattern (BrokerGateway -> RiskGate -> OrderManager) follows established trading system architecture. Risk-as-middleware is well-documented best practice. Custom TWAP for thin markets is a pragmatic choice given IB's algo limitations.
- Pitfalls: HIGH - IB disconnection behavior, client ID conflicts, and paper fill optimism are well-documented in community and official sources. Port confusion is a known operational risk with documented mitigations.

**Research date:** 2026-02-19
**Valid until:** 2026-04-19 (60 days -- ib_async and IBC are stable releases, IB API changes infrequently)
