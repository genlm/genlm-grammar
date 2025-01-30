from setuptools import setup, find_packages

requirements = [
    "frozendict",
    "interegular",
    "lark",
    "nltk",
    "numpy",
    "arsenal @ git+https://github.com/timvieira/arsenal",
    "IPython",
    "graphviz",
    "pandas",
]

test_requirements = [
    "pytest",
    "pytest-benchmark",
]

docs_requirements = [
    "mkdocs",
    "mkdocstrings[python]",
    "mkdocs-material",
    "mkdocs-gen-files",
    "mkdocs-literate-nav",
    "mkdocs-section-index",
]

setup(
    name="genlm-grammar",
    version="0.0.1",
    description="",
    install_requires=requirements,
    extras_require={"test": test_requirements, "docs": docs_requirements},
    python_requires=">=3.10",
    authors=["The GenLM Team"],
    readme="README.md",
    scripts=[],
    packages=find_packages(),
)
