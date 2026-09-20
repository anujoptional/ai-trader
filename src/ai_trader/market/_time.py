"""Minute bucketing shared by the market-data aggregation modules.

Candle building and volume differencing must agree exactly on where a minute
starts, so that logic lives here rather than being repeated per module.
"""

from datetime import datetime, timedelta

ONE_MINUTE = timedelta(minutes=1)


def minute_start(timestamp: datetime) -> datetime:
    """Return the start of the minute containing ``timestamp``."""
    return timestamp.replace(second=0, microsecond=0)


__all__ = ["ONE_MINUTE", "minute_start"]
