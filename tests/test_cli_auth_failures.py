from importlib import import_module
from inspect import signature
from unittest.mock import patch

import pytest
from pytest import CaptureFixture

from ai_trader.broker.groww import GrowwAuthenticationError

_CLI_MODULES = (
    "check_features",
    "check_groww",
    "check_historical_data",
    "check_market_data",
    "check_market_state",
    "check_stream",
)


@pytest.mark.parametrize("module_name", _CLI_MODULES)
def test_every_cli_reports_an_authentication_failure_as_its_own_cause(
    module_name: str,
    capsys: CaptureFixture[str],
) -> None:
    # GrowwAuthenticationError is a GrowwBrokerError, so a CLI that catches only
    # the base class blames its own operation for a failure that happened before
    # that operation began, sending an operator to debug the wrong layer.
    module = import_module(f"ai_trader.cli.{module_name}")

    with (
        patch(f"ai_trader.cli.{module_name}.load_groww_settings"),
        patch(
            f"ai_trader.cli.{module_name}.GrowwBroker.authenticate",
            side_effect=GrowwAuthenticationError("test-sensitive-detail"),
        ),
    ):
        # check_features is the only one that parses argv; pass an empty list so
        # it cannot inherit pytest's own arguments.
        main = module.main
        exit_code = main([]) if signature(main).parameters else main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "Groww authentication failed.\n"
    assert "test-sensitive-detail" not in captured.err
