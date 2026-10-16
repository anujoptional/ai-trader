"""The session timeline shared by the live run and replays: when the agent takes its
turns, the 15:10 close and the day's report."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime, time, timedelta
from typing import Any

from ai_trader.agent.account import PaperAccount
from ai_trader.agent.brain import CLOSE_AT, OPEN_AT, AgentTools, AutonomousAgent

FIRST_TURN = time(9, 0)
LAST_TURN = time(15, 7)
OPENING_FILL = time(9, 15, 20)
REVIEW = (
    "The session is over: everything still open was closed at 15:10 (see "
    "since_last_turn). Review the day in a few sentences: what you did, what worked, "
    "what did not, and what you would do differently. No more trading today; end "
    "with finish_turn."
)


class SessionRunner:
    """When the agent takes its turns: first at 09:00 (or at the start), then when it
    asked to look again, or at once after the opening fills or when an alert, stop
    or target fires; no turn starts at or after 15:07 or once the budget is spent."""

    def __init__(
        self,
        tools: AgentTools,
        agent: AutonomousAgent,
        log: Callable[[str], None],
        start: datetime,
    ) -> None:
        self.tools = tools
        self.agent = agent
        self.log = log
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.wake = threading.Event()
        first = datetime.combine(start.date(), FIRST_TURN, start.tzinfo)
        self.next_turn = max(start, first)
        self.why = "first turn of the day" + (
            ": pre-open, plan the session" if start.time() < OPEN_AT else ""
        )
        self.opened = start.time() >= OPENING_FILL

    def add_events(self, found: list[dict[str, Any]]) -> None:
        with self._lock:
            self.events.extend(found)
        self.wake.set()

    def watch(self, moment: datetime) -> None:
        """Between turns (another thread): stops, targets and alerts."""
        if not (OPEN_AT <= moment.time() < CLOSE_AT):
            return
        try:
            found = self.tools.watch()
        except Exception as error:
            self.log(f"watch failed: {type(error).__name__}: {error}")
            return
        for event in found:
            self.log(f"agent {event['event']}: {json.dumps(event, default=str)}")
        if found:
            self.add_events(found)

    def step(self, moment: datetime) -> float | None:
        """One pass: the opening fill, then a turn if one is due. Seconds to wait
        before the next pass; None at 15:10."""
        if moment.time() >= CLOSE_AT:
            return None
        if not self.opened and moment.time() >= OPENING_FILL:
            try:
                fills = self.tools.fill_opening()
            except Exception as error:
                self.log(f"opening fill failed ({type(error).__name__}: {error})")
                fills = None
            if fills is not None:
                self.opened = True
            if fills:
                self.log(f"opening fills: {json.dumps(fills, default=str)}")
                self.add_events([{"event": "opening_fills", "fills": fills}])
                self.next_turn, self.why = moment, "your opening orders were filled"
        if self.wake.is_set():
            self.wake.clear()
            if moment < self.next_turn:
                self.next_turn = moment
                self.why = "an alert, stop or target triggered"
        due = moment >= self.next_turn and moment.time() < LAST_TURN
        if not due or self.agent.budget_left() <= 0:
            return max(1.0, min((self.next_turn - moment).total_seconds(), 5.0))
        with self._lock:
            batch, self.events = self.events, []
        try:
            result = self.agent.turn(self.why, batch)
        except Exception as error:
            self.log(f"agent turn failed: {type(error).__name__}: {error}")
            with self._lock:
                self.events[:0] = batch
            self.next_turn = self.tools.now() + timedelta(minutes=2)
            self.why = "retry after a failed turn"
            return 0.0
        self.log(
            f"agent turn {self.agent.turns}: {result.steps} steps, "
            f"{len(result.searches)} searches, {len(result.trades)} trade actions, "
            f"${result.cost_usd:.2f} (day ${self.agent.spent:.2f})"
            + (f", stopped: {result.stopped}" if result.stopped else "")
        )
        self.log(f"agent notes: {result.notes[:1500]}")
        minutes = timedelta(minutes=result.next_check_minutes)
        self.next_turn = self.tools.now() + minutes
        self.why = "the time you asked to look again"
        if self.agent.budget_left() <= 0:
            self.log("agent AI budget used up: no more turns; exits still watched")
        return 0.0


def close_everything(
    tools: AgentTools,
    log: Callable[[str], None],
    attempts: int = 12,
    pause: float = 5.0,
) -> list[dict[str, Any]]:
    """Close every open position at market, retrying quote failures."""
    closed: list[dict[str, Any]] = []
    for attempt in range(attempts):
        try:
            closed += tools.close_all("time_1510")
        except Exception as error:
            log(f"15:10 close failed ({type(error).__name__}: {error})")
        if not tools.account.open_positions():
            break
        if attempt + 1 < attempts:
            threading.Event().wait(pause)
    log(f"15:10 close: {json.dumps(closed, default=str)}")
    left = tools.account.open_positions()
    if left:
        log(f"still open after the close: {[r['symbol'] for r in left]}")
    return closed


def report(account: PaperAccount, agent: AutonomousAgent) -> dict[str, Any]:
    summary = account.summary({})
    closed = summary["closed_today"]
    return {
        "model": agent.model,
        "effort": agent.effort,
        "turns": agent.turns,
        "ai_spend_usd": round(agent.spent, 2),
        "trades": len(closed),
        "won": sum(t["net_rs"] > 0 for t in closed),
        "gross_rs": round(sum(t["net_rs"] + t["charges_rs"] for t in closed), 2),
        "charges_rs": summary["charges_today_rs"],
        "net_rs": summary["realised_net_rs"],
        "equity_rs": summary["equity_rs"],
        "closed": closed,
    }
