"""Tests for pure Streamlit dashboard helpers."""

from __future__ import annotations

from app.streamlit_app import tab_order_for_mode


def test_app_modes_show_focused_workspaces_and_primary_tab_first():
    research = tab_order_for_mode("Research Mode")
    live = tab_order_for_mode("Live Mode")
    paper = tab_order_for_mode("Paper Trading Mode")

    assert research[:3] == ["Overview", "Backtest", "Model Comparison"]
    assert live[:3] == ["Live Market", "Overview", "Risk Simulator"]
    assert paper[:3] == ["Paper Trading", "Live Market", "Overview"]
    assert "Paper Trading" not in research
    assert "Backtest" not in live
    assert "Model Comparison" not in paper
    assert len(research) == len(set(research))
    assert len(live) == len(set(live))
    assert len(paper) == len(set(paper))
