"""Risk-budgeted generic and foreign-exchange position sizing."""

from __future__ import annotations

import math


def calculate_fx_position_size(
    account_size: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    pip_size: float = 0.0001,
    pip_value_per_standard_lot: float = 10.0,
    min_lot: float = 0.01,
) -> dict:
    """Calculate forex position size from stop distance measured in pips."""
    warnings = _input_warnings(account_size, risk_pct, entry_price, stop_price)
    if pip_size <= 0:
        warnings.append("pip_size must be positive.")
    if pip_value_per_standard_lot <= 0:
        warnings.append("pip_value_per_standard_lot must be positive.")
    if min_lot <= 0:
        warnings.append("min_lot must be positive.")
    stop_pips = (
        abs(entry_price - stop_price) / pip_size
        if pip_size > 0 and entry_price > 0 and stop_price > 0
        else 0.0
    )
    if math.isclose(stop_pips, 0):
        warnings.append("Entry and stop prices must define a non-zero pip distance.")
    risk_amount = account_size * risk_pct if account_size > 0 and risk_pct > 0 else 0.0
    standard_lots = (
        risk_amount / (stop_pips * pip_value_per_standard_lot)
        if not warnings and stop_pips > 0
        else 0.0
    )
    if 0 < standard_lots < min_lot:
        warnings.append(
            f"Calculated lot size {standard_lots:.4f} is below minimum lot {min_lot:.4f}."
        )
    valid = not warnings
    return {
        "asset_type": "fx",
        "account_size": float(account_size),
        "risk_pct": float(risk_pct),
        "risk_amount": float(risk_amount),
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "stop_distance": float(abs(entry_price - stop_price)),
        "stop_pips": float(stop_pips),
        "standard_lots": float(standard_lots),
        "mini_lots": float(standard_lots * 10),
        "micro_lots": float(standard_lots * 100),
        "lots": float(standard_lots),
        "units": float(standard_lots * 100_000),
        "estimated_loss_at_stop": float(
            standard_lots * stop_pips * pip_value_per_standard_lot
        ),
        "valid": valid,
        "warnings": warnings,
        "display_unit": "standard lots",
    }


def calculate_fixed_fractional_position_size(
    account_size: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
) -> dict:
    """Calculate generic fixed-fractional units from price risk."""
    warnings = _input_warnings(account_size, risk_pct, entry_price, stop_price)
    stop_distance = (
        abs(entry_price - stop_price)
        if entry_price > 0 and stop_price > 0
        else 0.0
    )
    if math.isclose(stop_distance, 0):
        warnings.append("Entry and stop prices must differ.")
    risk_amount = account_size * risk_pct if account_size > 0 and risk_pct > 0 else 0.0
    units = risk_amount / stop_distance if not warnings and stop_distance > 0 else 0.0
    return {
        "asset_type": "generic",
        "account_size": float(account_size),
        "risk_pct": float(risk_pct),
        "risk_amount": float(risk_amount),
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "stop_distance": float(stop_distance),
        "units": float(units),
        "lots": 0.0,
        "notional": float(units * entry_price),
        "estimated_loss_at_stop": float(units * stop_distance),
        "valid": not warnings,
        "warnings": warnings,
        "display_unit": "units",
    }


def _input_warnings(
    account_size: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
) -> list[str]:
    warnings: list[str] = []
    if account_size <= 0:
        warnings.append("account_size must be positive.")
    if risk_pct <= 0:
        warnings.append("risk_pct must be positive.")
    if risk_pct > 1:
        warnings.append("risk_pct must not exceed 1.")
    if entry_price <= 0:
        warnings.append("entry_price must be positive.")
    if stop_price <= 0:
        warnings.append("stop_price must be positive.")
    return warnings
