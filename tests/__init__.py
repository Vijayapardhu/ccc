"""Test package.

Present so test modules can use relative imports for the shared synthetic-data
builders in ``conftest``. Without it pytest imports each test module as a
top-level module and ``from .conftest import ...`` fails at collection.
"""
