"""On-disk cache of historical candles, so a replay can be re-run.

The layer exists to separate *getting* the bars from *reasoning about* them.
``ai_trader.replay`` takes an iterable of candles and says nothing about where
they came from; this package is one answer to that, and the one that makes a
published result reproducible, because it can serve a range with no broker
credentials at all.
"""

from ai_trader.history.store import (
    CandleStore,
    CandleStoreError,
    last_completed_session_close,
)

__all__ = ["CandleStore", "CandleStoreError", "last_completed_session_close"]
