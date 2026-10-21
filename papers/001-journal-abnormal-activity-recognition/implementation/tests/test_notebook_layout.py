"""Offline checks for numbered notebooks and their standalone Python exports."""

import ast
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT / "notebooks"


def test_all_notebooks_have_unique_ordered_short_names():
    notebooks = sorted(BOOKS.glob("*.ipynb"))
    assert len(notebooks) == 10
    assert all(re.fullmatch(r"\d{2}_[a-z0-9_]+", book.stem) for book in notebooks)
    assert [int(book.stem[:2]) for book in notebooks] == list(range(1, 11))


def test_numbered_exports_live_in_scripts_and_match_notebook_code():
    assert not [path for path in BOOKS.glob("*.py") if path.name != "__init__.py"]
    exports = sorted((BOOKS / "scripts").glob("*.py"))
    assert len(exports) == 6
    for export in exports:
        assert re.fullmatch(r"\d{2}_[a-z0-9_]+", export.stem), export
        notebook_path = BOOKS / f"{export.stem}.ipynb"
        assert notebook_path.is_file(), export
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert ast.dump(ast.parse(export.read_text(encoding="utf-8"))) == ast.dump(
            ast.parse(code)
        ), export
