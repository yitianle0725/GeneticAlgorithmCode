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


def build_graph(dpi: float = 72.45) -> Digraph:
    graph = Digraph("IEMOEC_framework", engine="dot")
    graph.attr(
        rankdir="LR",
        bgcolor="white",
        fontname="Helvetica",
        fontsize="10",
        pad="0.08",
        margin="0",
        nodesep="0.22",
        ranksep="0.42",
        splines="ortho",
        outputorder="edgesfirst",
        size="7.16,4.15!",
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
        margin="0.08,0.05",
        height="0.34",
    )
    graph.attr(
        "edge",
        fontname="Helvetica",
        fontsize="8.5",
        fontcolor=DARK,
        color=DARK,
        arrowsize="0.62",
        penwidth="0.9",
    )

    graph.node("start", "Start", shape="ellipse", style="filled", fillcolor="white")
    add_process(
        graph,
        "initialize",
        "Initialize origin population P_o\n"
        "Float random sampling + evaluation\n"
        "A <- P_o",
    )

    with graph.subgraph(name="cluster_independent") as independent:
        independent.attr(
            label="1  INDEPENDENT SUBPOPULATION EVOLUTION",
            labelloc="t",
            labeljust="l",
            fontname="Helvetica-Bold",
            fontsize="10",
            color=DARK,
            penwidth="1.1",
            style="rounded",
            margin="10",
        )
        independent.node(
            "phase_check",
            "FE / MaxFEs\n< 0.40?",
            shape="diamond",
            style="filled",
            fillcolor=BLUE,
            color=BLUE_BORDER,
            penwidth="1.15",
            margin="0.03",
        )
        add_process(
            independent,
            "aggregation",
            "YES - Aggregation Phase\n"
            "Tchebycheff local survival\n"
            "recombination ratio = 1.0",
            accent=True,
        )
        add_process(
            independent,
            "pareto",
            "NO - Pareto Phase\n"
            "nondominated fronts + weight tie-break\n"
            "recombination ratio = 0.25",
            accent=True,
        )
        add_process(
            independent,
            "define_islands",
            "Build 2M independent islands each round\n"
            "M axial weights + M random aggregation weights\n"
            "select one origin ancestor per island",
        )
        add_process(
            independent,
            "expand",
            "Single-ancestor island expansion\n"
            "clone selected origin + Polynomial Mutation\n"
            "evaluate mutated copies",
        )
        independent.node(
            "island_bank",
            "{2M ISLANDS (representative view)|"
            "{Island 1|Island 2|...|Island 2M}}",
            shape="record",
            style="filled",
            fillcolor=LIGHT_GRAY,
            color=DARK,
            penwidth="1.0",
        )
        add_process(
            independent,
            "local_evolution",
            "Independent evolution in every island\n"
            "one generation: random pairing -> SBX -> Polynomial Mutation\n"
            "evaluation -> phase-specific local survival",
        )

        with independent.subgraph() as phase_column:
            phase_column.attr(rank="same")
            phase_column.node("phase_check")
            phase_column.node("aggregation")
            phase_column.node("pareto")
        with independent.subgraph() as evolution_column:
            evolution_column.attr(rank="same")
            evolution_column.node("define_islands")
            evolution_column.node("expand")
            evolution_column.node("island_bank")
            evolution_column.node("local_evolution")

        independent.edge(
            "phase_check",
            "aggregation",
            style="dotted",
            color=MID_GRAY,
            constraint="false",
        )
        independent.edge(
            "phase_check",
            "pareto",
            style="dotted",
            color=MID_GRAY,
            constraint="false",
        )
        independent.edge(
            "aggregation",
            "define_islands",
            style="dotted",
            color=MID_GRAY,
        )
        independent.edge(
            "pareto",
            "define_islands",
            style="dotted",
            color=MID_GRAY,
        )
        independent.edge("define_islands", "expand", constraint="false")
        independent.edge("expand", "island_bank", constraint="false")
        independent.edge("island_bank", "local_evolution", constraint="false")

    with graph.subgraph(name="cluster_combination") as combination:
        combination.attr(
            label="2  CROSS-ISLAND EXTREMA COMBINATION",
            labelloc="t",
            labeljust="l",
            fontname="Helvetica-Bold",
            fontsize="10",
            color=MID_GRAY,
            penwidth="1.2",
            style="rounded,dashed",
            margin="10",
        )
        add_process(
            combination,
            "extract_extrema",
            "Extract one local extremum per island\n"
            "minimum weighted Tchebycheff score",
        )
        add_process(
            combination,
            "direction_pairing",
            "Pair extrema from least-similar\n"
            "normalized weight directions\n"
            "up to 2 partners per extremum",
        )
        add_process(
            combination,
            "combination_offspring",
            "Extrema Combination\n"
            "SBX + Polynomial Mutation + evaluation\n"
            "phase budget: 1.0 (Aggregation) / 0.25 (Pareto)",
        )
        with combination.subgraph() as combination_column:
            combination_column.attr(rank="same")
            combination_column.node("extract_extrema")
            combination_column.node("direction_pairing")
            combination_column.node("combination_offspring")
        combination.edge("extract_extrema", "direction_pairing", constraint="false")
        combination.edge(
            "direction_pairing",
            "combination_offspring",
            constraint="false",
        )

    with graph.subgraph(name="cluster_global") as global_selection:
        global_selection.attr(
            label="3  GLOBAL REFERENCE-DIRECTION ENVIRONMENTAL SELECTION",
            labelloc="t",
            labeljust="l",
            fontname="Helvetica-Bold",
            fontsize="10",
            color=BLUE_BORDER,
            penwidth="1.5",
            style="rounded",
            margin="10",
        )
        add_process(
            global_selection,
            "merge",
            "Merge candidate solutions\n"
            "A + P_o + all island populations\n"
            "+ extrema-combination offspring",
        )
        add_process(
            global_selection,
            "archive_selection",
            "NSGA-III ReferenceDirectionSurvival\n"
            "A <- global candidate pool (size N)",
            accent=True,
        )
        add_process(
            global_selection,
            "origin_selection",
            "NSGA-III ReferenceDirectionSurvival\n"
            "P_o <- select origin from A\n"
            "|P_o| = min(N, max(20, ceil(0.2N)))",
            accent=True,
        )
        global_selection.node(
            "termination",
            "MaxFEs reached?\nNO: next outer iteration",
            shape="diamond",
            style="filled",
            fillcolor="white",
            color=DARK,
            margin="0.03",
        )
        add_process(
            global_selection,
            "final_population",
            "YES - Final reference-direction selection\nReturn final population",
        )
        with global_selection.subgraph() as global_column:
            global_column.attr(rank="same")
            global_column.node("merge")
            global_column.node("archive_selection")
            global_column.node("origin_selection")
            global_column.node("termination")
            global_column.node("final_population")
        global_selection.edge("merge", "archive_selection", constraint="false")
        global_selection.edge(
            "archive_selection",
            "origin_selection",
            constraint="false",
        )
        global_selection.edge("origin_selection", "termination", constraint="false")
        global_selection.edge("termination", "final_population", constraint="false")

    with graph.subgraph() as initialization_column:
        initialization_column.attr(rank="same")
        initialization_column.node("start")
        initialization_column.node("initialize")
    graph.edge(
        "start",
        "initialize",
        style="dotted",
        color=MID_GRAY,
        constraint="false",
    )
    graph.edge(
        "initialize",
        "define_islands",
        constraint="false",
    )
    graph.edge(
        "initialize",
        "phase_check",
        style="dotted",
        color=MID_GRAY,
        constraint="false",
    )
    graph.edge(
        "local_evolution",
        "extract_extrema",
    )
    graph.edge("local_evolution", "merge", constraint="false")
    graph.edge(
        "combination_offspring",
        "merge",
    )
    graph.edge(
        "termination",
        "phase_check",
        style="dashed",
        color=BLUE_BORDER,
        fontcolor=BLUE_BORDER,
        penwidth="1.15",
        constraint="false",
    )

    graph.attr(
        label="LEGEND   solid arrow: candidate-solution flow   |   "
        "dotted arrow: phase/control signal   |   "
        "dashed arrow: outer-loop feedback",
        labelloc="b",
        labeljust="c",
        fontsize="8.5",
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
