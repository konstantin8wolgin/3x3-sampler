"""Offline access to deterministic reports already stored in completed runs."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _render_svg(run: dict[str, Any], estimate: dict[str, Any]) -> str:
    value = estimate["value"]
    label = html.escape(str(run["observable"]))
    shown = f"{value['real']}{value['imag']:+}j"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="150" '
        'viewBox="0 0 960 150">'
        '<rect width="100%" height="100%" fill="#f8fafc"/>'
        '<text x="32" y="45" font-family="system-ui,sans-serif" font-size="22" '
        'fill="#18222c">Taylor–Gauss 3×3 stored estimate</text>'
        f'<text x="42" y="100" font-family="ui-monospace,monospace" font-size="17" '
        f'fill="#334155">{label} = {html.escape(shown)}</text></svg>\n'
    )


def _render_html(run: dict[str, Any], estimate: dict[str, Any]) -> str:
    value = estimate["value"]
    uncertainty = (
        "analytic; not applicable"
        if estimate["standard_error_real"] is None
        else (
            f"SE(real)={estimate['standard_error_real']}; "
            f"SE(imag)={estimate['standard_error_imag']}"
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Taylor–Gauss 3×3 run report</title></head><body>
<h1>Taylor–Gauss 3×3 run report</h1>
<p><strong>Authority:</strong> {html.escape(str(run['authority']))}</p>
<p><strong>Method:</strong> {html.escape(str(run['method']))}</p>
<p><strong>Observable:</strong> {html.escape(str(run['observable']))}</p>
<p><strong>Estimate:</strong> {value['real']}{value['imag']:+}j</p>
<p><strong>Uncertainty:</strong> {html.escape(uncertainty)}</p>
<img src="figures/estimates.svg" alt="Stored estimate">
<p>This offline report uses only completed stored content.</p>
</body></html>
"""


def render_report(source: str | Path, output: str | Path | None = None) -> Path:
    """Return the stored report or copy it into a new immutable derivative run."""

    from .artifacts import _copy_derivative, _load_completed

    source_path, _, _, _ = _load_completed(source)
    if output is None:
        return source_path / "report.html"
    return _copy_derivative(source_path, Path(output), "report")


__all__ = ["render_report"]
