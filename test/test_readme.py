import pathlib

import exdown
import pytest

this_dir = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "string,lineno",
    exdown.extract(this_dir.parent / "README.md", syntax_filter="python"),
)
def test_readme(string, lineno):
    try:
        # https://stackoverflow.com/a/62851176/353337
        exec(string, {"__MODULE__": "__main__"})
    except Exception:
        print(f"README.md (line {lineno}):\n```\n{string}```")
        raise
