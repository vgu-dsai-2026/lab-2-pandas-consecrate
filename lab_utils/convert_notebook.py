from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a notebook to Python with nbconvert, then keep only imports, "
            "top-level constant assignments, and function definitions."
        )
    )
    parser.add_argument(
        "notebook",
        nargs="?",
        default="notebook.ipynb",
        help="Path to the notebook file to convert.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the filtered Python output file. Defaults to <notebook>.py.",
    )
    return parser.parse_args()


def export_notebook(notebook_path: Path, output_path: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        normalized_notebook_path = temp_path / notebook_path.name
        normalize_notebook(notebook_path, normalized_notebook_path)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "nbconvert",
                "--to",
                "python",
                str(normalized_notebook_path),
                "--output",
                output_path.stem,
                "--output-dir",
                str(output_path.parent),
            ],
            check=True,
            cwd=temp_dir,
        )


def normalize_notebook(notebook_path: Path, output_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)

    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "execute_result" and "execution_count" not in output:
                output["execution_count"] = None

    nbformat.write(notebook, output_path)


def is_literal_assignment(node: ast.Assign | ast.AnnAssign) -> bool:
    value = node.value
    if value is None:
        return False

    try:
        ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return False

    return True


def filter_python(source: str) -> str:
    tree = ast.parse(source)
    kept_nodes: list[ast.stmt] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef)):
            kept_nodes.append(node)
            continue

        if isinstance(node, (ast.Assign, ast.AnnAssign)) and is_literal_assignment(node):
            kept_nodes.append(node)

    filtered_tree = ast.Module(body=kept_nodes, type_ignores=[])
    ast.fix_missing_locations(filtered_tree)
    return ast.unparse(filtered_tree) + "\n"


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook).resolve()
    output_path = Path(args.output).resolve() if args.output else notebook_path.with_suffix(".py")

    export_notebook(notebook_path, output_path)
    filtered_source = filter_python(output_path.read_text(encoding="utf-8"))
    output_path.write_text(filtered_source, encoding="utf-8")

    print(f"Filtered notebook written to {output_path}")


if __name__ == "__main__":
    main()
