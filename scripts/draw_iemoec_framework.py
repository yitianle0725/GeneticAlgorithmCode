#!/usr/bin/env python
"""Generate the paper-ready IEMOEC framework diagram with Graphviz."""

from __future__ import annotations

from pathlib import Path

from graphviz import Digraph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "pictures" / "iemoec_framework"
FILE_STEM = "IEMOEC_framework"

BLUE = "#DDEAF2"
BLUE_BORDER = "#557A95"
DARK = "#2F3437"
MID_GRAY = "#6B7074"
LIGHT_GRAY = "#F4F5F6"


def add_process(graph: Digraph, name: str, label: str, *, accent: bool = False) -> None:
    """Add one consistently styled process node."""
    graph.node(
        name,
        label,
        shape="box",
        style="rounded,filled",
        fillcolor=BLUE if accent else "white",
        color=BLUE_BORDER if accent else DARK,
        penwidth="1.15" if accent else "0.9",
    )


def add_cluster_title(cluster, label: str, *, color: str, style: str) -> None:
    """Apply the shared publication style to one algorithm layer."""
    cluster.attr(
        label=label,
        labelloc="t",
        labeljust="l",
        fontname="Helvetica-Bold",
        fontsize="9.5",
        color=color,
        penwidth="1.1",
        style=style,
        margin="4",
    )


def build_graph(dpi: float = 72.0) -> Digraph:
    graph = Digraph("IEMOEC_framework", engine="dot")
    graph.attr(
        rankdir="TB",
        bgcolor="white",
        fontname="Helvetica",
        fontsize="9.5",
        pad="0",
        margin="0",
        nodesep="0.115",
        ranksep="0.02",
        splines="ortho",
        outputorder="edgesfirst",
        dpi=str(dpi),
        compound="true",
        newrank="true",
    )
    graph.attr(
        "node",
        fontname="Helvetica",
        fontsize="9.5",
        fontcolor=DARK,
        color=DARK,
        penwidth="0.9",
        margin="0.05,0.035",
        height="0.30",
    )
    graph.attr(
        "edge",
        fontname="Helvetica",
        fontsize="9",
        fontcolor=DARK,
        color=DARK,
        arrowsize="0.58",
        penwidth="0.9",
    )

    graph.node("start", "Start", shape="ellipse", style="filled", fillcolor="white")
    add_process(
        graph,
        "initialize",
        "Initialize P_o\nEvaluate\nA <- P_o",
    )

    with graph.subgraph(name="cluster_independent") as independent:
        add_cluster_title(
            independent,
            "1  INDEPENDENT\nSUBPOPULATION\nEVOLUTION",
            color=DARK,
            style="rounded",
        )
        independent.node(
            "phase",
            "Phase control\nFE ratio\n< 0.40?",
            shape="diamond",
            style="filled",
            fillcolor=BLUE,
            color=BLUE_BORDER,
            penwidth="1.15",
            margin="0.02",
        )
        add_process(
            independent,
            "build_islands",
            "Periodic multi-ancestor\nconstruction from A (0 FE)",
        )
        independent.node(
            "island_bank",
            "{2M ISLANDS|{M axial|M random}|"
            "direction elite + neighbors|+ diverse solutions}",
            shape="record",
            style="filled",
            fillcolor=LIGHT_GRAY,
            color=DARK,
            penwidth="1.0",
        )
        add_process(
            independent,
            "local_evolution",
            "Independent SBX + PM\nEvaluate\nTCH / Pareto survival",
            accent=True,
        )

        independent.edge("phase", "build_islands", style="invis", weight="5")
        independent.edge("build_islands", "island_bank")
        independent.edge("island_bank", "local_evolution")
        independent.edge(
            "phase",
            "local_evolution",
            style="dotted",
            color=MID_GRAY,
            fontcolor=MID_GRAY,
            constraint="false",
        )

    with graph.subgraph(name="cluster_combination") as combination:
        add_cluster_title(
            combination,
            "2  CROSS-ISLAND\nEXTREMA\nCOMBINATION",
            color=MID_GRAY,
            style="rounded,dashed",
        )
        add_process(combination, "extrema", "One weighted\nextremum per island")
        add_process(combination, "pairing", "Least-similar direction\npairing")
        add_process(
            combination,
            "combination_offspring",
            "SBX + PM + evaluation\nQ_c; budget 1.0 / 0.25",
            accent=True,
        )

        combination.edge("extrema", "pairing")
        combination.edge("pairing", "combination_offspring")

    with graph.subgraph(name="cluster_global") as global_selection:
        add_cluster_title(
            global_selection,
            "3  GLOBAL\nREFERENCE-DIRECTION\nSELECTION",
            color=BLUE_BORDER,
            style="rounded",
        )
        add_process(
            global_selection,
            "merge",
            "Merge\nA + P_o\n+ islands + Q_c",
        )
        add_process(
            global_selection,
            "archive_selection",
            "NSGA-III survival\nA <- select N",
            accent=True,
        )
        add_process(
            global_selection,
            "origin_selection",
            "NSGA-III survival\nP_o <- select origins",
            accent=True,
        )
        global_selection.node(
            "termination",
            "MaxFEs\nreached?",
            shape="diamond",
            style="filled",
            fillcolor="white",
            color=DARK,
            margin="0.02",
        )
        add_process(
            global_selection,
            "final_population",
            "Final NSGA-III survival\nReturn population",
        )

        global_selection.edge("merge", "archive_selection")
        global_selection.edge("archive_selection", "origin_selection")
        global_selection.edge("origin_selection", "termination")
        global_selection.edge(
            "termination",
            "final_population",
            xlabel="YES",
            minlen="2",
        )

    graph.edge("start", "initialize")
    with graph.subgraph() as entry_row:
        entry_row.attr(rank="same")
        entry_row.node("initialize")
        entry_row.node("build_islands")
        entry_row.node("extrema")
        entry_row.node("merge")
    graph.edge("initialize", "build_islands", constraint="false")
    graph.edge(
        "phase",
        "combination_offspring",
        style="dotted",
        color=MID_GRAY,
        fontcolor=MID_GRAY,
        constraint="false",
    )
    graph.edge("local_evolution", "extrema")
    graph.edge("local_evolution", "merge", constraint="false")
    graph.edge("combination_offspring", "merge")
    graph.edge(
        "termination",
        "build_islands",
        xlabel="NO",
        style="dashed",
        color=BLUE_BORDER,
        fontcolor=BLUE_BORDER,
        penwidth="1.15",
        constraint="false",
    )

    graph.attr(
        label="LEGEND   solid: candidate flow   |   dotted: phase control   |   "
        "dashed: outer-loop feedback",
        labelloc="b",
        labeljust="c",
        fontsize="9",
        fontcolor=MID_GRAY,
    )
    return graph


def render_all(graph: Digraph) -> None:
    """Save the DOT source and render editable and publication formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dot_path = OUTPUT_DIR / f"{FILE_STEM}.dot"
    dot_path.write_text(graph.source, encoding="utf-8")

    for output_format in ("svg", "pdf"):
        graph.render(
            filename=FILE_STEM,
            directory=OUTPUT_DIR,
            format=output_format,
            cleanup=True,
        )

    preview_graph = build_graph(dpi=240)
    preview_graph.render(
        filename=f"{FILE_STEM}_preview",
        directory=OUTPUT_DIR,
        format="png",
        cleanup=True,
    )


def main() -> None:
    render_all(build_graph())
    print(f"IEMOEC framework files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
