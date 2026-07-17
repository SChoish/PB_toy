from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import Episode
from .env import ToyEnv


def _segment_hazard_hit(
    env: ToyEnv, a: np.ndarray, b: np.ndarray
) -> tuple[bool, np.ndarray | None]:
    """Return (hits, point) for the first hazard intersection along a→b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    denom = float(d @ d)
    best_t, best_p = None, None
    for h in env.hazards:
        c = np.asarray(h.center, float)
        r = h.radius + env.clearance
        # Solve |a + t d - c|^2 = r^2
        f = a - c
        qa = denom
        qb = 2.0 * float(f @ d)
        qc = float(f @ f) - r * r
        if qa < 1e-12:
            continue
        disc = qb * qb - 4 * qa * qc
        if disc < 0:
            continue
        sqrt_disc = np.sqrt(disc)
        for t in ((-qb - sqrt_disc) / (2 * qa), (-qb + sqrt_disc) / (2 * qa)):
            if 0.0 <= t <= 1.0:
                if best_t is None or t < best_t:
                    best_t = t
                    best_p = a + t * d
    return best_p is not None, best_p


def render_svg(
    path: str | Path,
    env: ToyEnv,
    episodes: list[Episode],
    start: np.ndarray,
    endpoint: np.ndarray,
    bridge_path: np.ndarray,
    support_windows: np.ndarray,
    metrics: dict,
) -> None:
    """Three-panel SVG focused on why an explicit bridge is useful."""
    W, H = 1200, 520
    panels = [(38, 112, 350, 330), (425, 112, 350, 330), (812, 112, 350, 330)]

    def xy(p: np.ndarray, panel: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, w, h = panel
        return x + float(p[0]) * w, y + (1.0 - float(p[1])) * h

    def poly(points: np.ndarray, panel, color: str, width: float, opacity=1.0, dash="") -> str:
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in (xy(p, panel) for p in points))
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"{extra}/>'
        )

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    out.append(
        """<defs>
<pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(35)">
  <line x1="0" y1="0" x2="0" y2="8" stroke="#b83b2e" stroke-width="2" opacity=".55"/>
</pattern>
<filter id="shadow"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity=".12"/></filter>
</defs>"""
    )
    out.append('<rect width="1200" height="520" fill="#fbfaf7"/>')
    out.append(
        '<text x="38" y="34" font-family="Arial,sans-serif" font-size="23" font-weight="700" '
        'fill="#18212b">Why the bridge matters</text>'
    )
    out.append(
        '<text x="38" y="58" font-family="Arial,sans-serif" font-size="14" fill="#5d6873">'
        "A selected endpoint says where to go—not how to traverse unsupported space</text>"
    )
    labels = [
        "A  Offline trajectory support",
        "B  Endpoint only",
        "C  Endpoint-pinned bridge",
    ]
    for panel, label in zip(panels, labels):
        x, y, w, h = panel
        out.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" '
            f'stroke="#cbd1d6" stroke-width="1.5" filter="url(#shadow)"/>'
        )
        out.append(
            f'<text x="{x}" y="{y-16}" font-family="Arial,sans-serif" font-size="16" '
            f'font-weight="700" fill="#24303b">{label}</text>'
        )
        for hz in env.hazards:
            cx, cy = xy(np.asarray(hz.center), panel)
            r = hz.radius * w
            out.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="#e76551" '
                f'fill-opacity=".18" stroke="#b83b2e" stroke-width="2"/>'
            )
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="url(#hatch)"/>')

    # Panel A: denser support (nearest windows + a few nearby episode slices)
    for window in support_windows:
        out.append(poly(window, panels[0], "#25282b", 2.1, 0.28))
    if len(support_windows):
        out.append(poly(support_windows[0], panels[0], "#25282b", 3.5, 0.88))

    # Panel B: straight chord + X at actual hazard hit
    out.append(poly(np.vstack([start, endpoint]), panels[1], "#2d78c4", 4.8, 1.0))
    hits, hit_p = _segment_hazard_hit(env, start, endpoint)
    if hits and hit_p is not None:
        mx, my = xy(hit_p, panels[1])
    else:
        mx, my = xy((start + endpoint) / 2, panels[1])
    out.append(
        f'<line x1="{mx-10:.1f}" y1="{my-10:.1f}" x2="{mx+10:.1f}" y2="{my+10:.1f}" '
        f'stroke="#b83b2e" stroke-width="4.2" stroke-linecap="round"/>'
    )
    out.append(
        f'<line x1="{mx+10:.1f}" y1="{my-10:.1f}" x2="{mx-10:.1f}" y2="{my+10:.1f}" '
        f'stroke="#b83b2e" stroke-width="4.2" stroke-linecap="round"/>'
    )
    out.append(
        f'<text x="{mx:.1f}" y="{my-16:.1f}" text-anchor="middle" font-family="Arial,sans-serif" '
        f'font-size="12" font-weight="700" fill="#b83b2e">unsafe</text>'
    )

    # Panel C: solid prefix + dashed remainder
    split = max(2, int(round(0.48 * (len(bridge_path) - 1))))
    out.append(poly(bridge_path[: split + 1], panels[2], "#4f2583", 5.6))
    out.append(poly(bridge_path[split:], panels[2], "#7b52a7", 4.2, 1.0, "7 6"))
    rx, ry = xy(bridge_path[split], panels[2])
    out.append(
        f'<circle cx="{rx:.1f}" cy="{ry:.1f}" r="5.5" fill="#4f2583" stroke="#fff" stroke-width="1.6"/>'
    )
    out.append(
        f'<text x="{rx+10:.1f}" y="{ry-8:.1f}" font-family="Arial,sans-serif" font-size="11" '
        f'fill="#4f2583">replan</text>'
    )

    for panel in panels:
        sx, sy = xy(start, panel)
        zx, zy = xy(endpoint, panel)
        out.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6.8" fill="#fff" stroke="#18212b" stroke-width="2.4"/>'
        )
        out.append(
            f'<circle cx="{zx:.1f}" cy="{zy:.1f}" r="7.8" fill="#fff" stroke="#6c3eb6" stroke-width="2.8"/>'
        )

    out.append(
        '<text x="600" y="478" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" '
        'fill="#4b5661">same start  ●   same selected endpoint  ○</text>'
    )
    stat = (
        f"dataset collisions: {metrics['unsafe_transitions']}  ·  "
        f"direct path: {'unsafe' if metrics.get('concept_direct_path_unsafe', True) else 'safe'}  ·  "
        f"learned pinned bridge: {'safe' if metrics.get('concept_pinned_bridge_safe', True) else 'unsafe'}"
    )
    out.append(
        f'<text x="600" y="502" text-anchor="middle" font-family="Arial,sans-serif" '
        f'font-size="12" fill="#4b5661">{stat}</text>'
    )
    out.append("</svg>")
    Path(path).write_text("\n".join(out), encoding="utf-8")
