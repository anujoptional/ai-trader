"""Broker-independent market-data aggregation."""

# Where a session starts is shared vocabulary rather than an implementation
# detail: the feature layer needs the same 09:15 this package buckets volume
# against, and a second definition of it would be a second answer to where one
# session ends and the next begins.
from ai_trader.market._time import (
    INDIA_TIMEZONE,
    SESSION_OPEN_TIME,
    trading_session_date,
)
from ai_trader.market.candles import (
    Candle,
    CandleBuilder,
    InvalidTickError,
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
    "INDIA_TIMEZONE",
    "SESSION_OPEN_TIME",
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
    "trading_session_date",
]
