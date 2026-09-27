"""Broker-independent market-data aggregation."""

# Clock names are not re-exported here. This package once owned them, so passing
# them through was the courtesy of the module that held the definition; now that
# ``ai_trader.clock`` holds it, a second import path is just a second name for
# one thing, and the question "where does 09:15 come from" would have two true
# answers again. Import them from ``ai_trader.clock``.
from ai_trader.market.candles import (
    Candle,
    CandleBuilder,
    InvalidTickError,
    to_candle,
)
from ai_trader.market.state import (
    InstrumentState,
    MarketState,
)

# The supervisor's tuning constants stay in ai_trader.market.stream rather than
# being re-exported here. "Session" means one bounded collect call there and the
# 09:15-15:30 trading day everywhere else in this package, and a flat
# DEFAULT_SESSION_SECONDS beside DEFAULT_POLL_INTERVAL_SECONDS would read as the
# latter.
from ai_trader.market.stream import (
    ClosableTickStream,
    StreamReport,
    StreamStopReason,
    StreamSupervisor,
    StreamSupervisorError,
    TickStream,
)
from ai_trader.market.volume import (
    CumulativeVolumeSnapshot,
    CumulativeVolumeTracker,
    MinuteVolume,
    VolumeEnricher,
)
from ai_trader.market.volume_poller import (
    DEFAULT_MAX_READING_AGE_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    VolumePoller,
    VolumePollerError,
)

__all__ = [
    "DEFAULT_MAX_READING_AGE_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "Candle",
    "CandleBuilder",
    "ClosableTickStream",
    "CumulativeVolumeSnapshot",
    "CumulativeVolumeTracker",
    "InstrumentState",
    "InvalidTickError",
    "MarketState",
    "MinuteVolume",
    "StreamReport",
    "StreamStopReason",
    "StreamSupervisor",
    "StreamSupervisorError",
    "TickStream",
    "VolumeEnricher",
    "VolumePoller",
    "VolumePollerError",
    "to_candle",
]
