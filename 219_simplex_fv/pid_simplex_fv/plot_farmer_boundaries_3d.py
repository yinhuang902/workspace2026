# plot_farmer_boundaries_3d.py
# ============================
# Interactive 3D Plotly visualization of pre-computed linear-region
# boundaries on the Farmer budget face (x_w + x_c + x_b = TOTAL).
#
# Usage (notebook):  paste this entire cell and run.
# Override boundary_json below if needed.

import json
from pathlib import Path

import plotly.graph_objects as go

# ── User-configurable ─────────────────────────────────────────────
BOUNDARY_JSON = "farmer_boundary_cache/boundaries_grid25.json"
# ──────────────────────────────────────────────────────────────────


def _resolve_boundary_path(rel_path):
    """Try cwd, then one level up (repo root), then script dir."""
    p = Path(rel_path)
    if p.exists():
        return p
    alt = Path.cwd().parent / rel_path
    if alt.exists():
        return alt
    # Try common workspace layout
    for parent in Path.cwd().parents:
        candidate = parent / rel_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot locate boundary file: {rel_path}\n"
        f"  cwd = {Path.cwd()}")


def plot_farmer_boundaries_3d(boundary_json=BOUNDARY_JSON):
    """Load boundaries and render an interactive 3D Plotly figure."""

    # ── Load JSON ─────────────────────────────────────────────────
    path = _resolve_boundary_path(boundary_json)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    TOTAL = data.get("TOTAL", data.get("total", 500.0))
    segments = data.get("segments", data.get("boundary_segments", []))

    print(f"Loaded {len(segments)} boundary segments from {path.name}")
    print(f"TOTAL = {TOTAL}")
    for i, seg in enumerate(segments[:3]):
        print(f"  seg[{i}]: {seg}")
    if len(segments) > 3:
        print(f"  ... ({len(segments) - 3} more)")

    # ── Convert 2D → 3D ──────────────────────────────────────────
    # Each segment: [[w1,c1],[w2,c2]]  →  (w, c, b=TOTAL-w-c)
    def to3d(pt):
        w, c = pt[0], pt[1]
        return (w, c, TOTAL - w - c)

    # ── Build boundary traces (batched with None separators) ──────
    CHUNK = 300
    boundary_traces = []
    for start in range(0, len(segments), CHUNK):
        chunk = segments[start:start + CHUNK]
        xs, ys, zs = [], [], []
        for seg in chunk:
            p1 = to3d(seg[0])
            p2 = to3d(seg[1])
            xs += [p1[0], p2[0], None]
            ys += [p1[1], p2[1], None]
            zs += [p1[2], p2[2], None]
        boundary_traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color="rgba(180,180,180,0.6)", width=2),
            name=f"boundaries ({start}-{start+len(chunk)-1})",
            showlegend=(start == 0),
            legendgroup="boundaries",
            hoverinfo="skip",
        ))

    # ── Budget face triangle outline ──────────────────────────────
    # Vertices: A=(0,0,TOTAL), B=(TOTAL,0,0), C=(0,TOTAL,0)
    tri_x = [0, TOTAL, 0, 0]
    tri_y = [0, 0, TOTAL, 0]
    tri_z = [TOTAL, 0, 0, TOTAL]
    face_trace = go.Scatter3d(
        x=tri_x, y=tri_y, z=tri_z,
        mode="lines",
        line=dict(color="rgba(80,80,80,0.9)", width=4),
        name="budget face",
    )

    # ── Budget face fill (semi-transparent) ───────────────────────
    face_mesh = go.Mesh3d(
        x=[0, TOTAL, 0],
        y=[0, 0, TOTAL],
        z=[TOTAL, 0, 0],
        i=[0], j=[1], k=[2],
        color="lightyellow",
        opacity=0.15,
        name="budget plane",
        showlegend=False,
        hoverinfo="skip",
    )

    # ── Feasible tetrahedron outline (faint context) ──────────────
    # Vertices: O=(0,0,0), A=(TOTAL,0,0), B=(0,TOTAL,0), C=(0,0,TOTAL)
    tet_edges = [
        ((0, 0, 0), (TOTAL, 0, 0)),
        ((0, 0, 0), (0, TOTAL, 0)),
        ((0, 0, 0), (0, 0, TOTAL)),
        ((TOTAL, 0, 0), (0, TOTAL, 0)),
        ((TOTAL, 0, 0), (0, 0, TOTAL)),
        ((0, TOTAL, 0), (0, 0, TOTAL)),
    ]
    tet_x, tet_y, tet_z = [], [], []
    for (x1, y1, z1), (x2, y2, z2) in tet_edges:
        tet_x += [x1, x2, None]
        tet_y += [y1, y2, None]
        tet_z += [z1, z2, None]
    tet_trace = go.Scatter3d(
        x=tet_x, y=tet_y, z=tet_z,
        mode="lines",
        line=dict(color="rgba(200,200,200,0.35)", width=1.5, dash="dot"),
        name="feasible tetrahedron",
        hoverinfo="skip",
    )

    # ── Assemble figure ───────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(face_mesh)
    fig.add_trace(tet_trace)
    fig.add_trace(face_trace)
    for tr in boundary_traces:
        fig.add_trace(tr)

    fig.update_layout(
        title=dict(
            text=(f"Farmer Linear-Region Boundaries on Budget Face "
                  f"(Σ = {TOTAL:.0f})<br>"
                  f"<sub>{path.name}  —  {len(segments)} segments</sub>"),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="wheat (acres)", range=[-10, TOTAL + 10]),
            yaxis=dict(title="corn (acres)",  range=[-10, TOTAL + 10]),
            zaxis=dict(title="beets (acres)", range=[-10, TOTAL + 10]),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
            bgcolor="white",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, t=60, b=0),
        width=900,
        height=750,
    )

    fig.show()
    return fig


# ── Run ───────────────────────────────────────────────────────────
fig = plot_farmer_boundaries_3d(BOUNDARY_JSON)
