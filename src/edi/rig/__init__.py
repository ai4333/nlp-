"""Rig builder — design an oil-rig P&ID by conversation, offline.

    chat  →  spec  →  layout  →  drawing  →  entities/relations  →  graph

`catalog` is what can be built, `templates` are the standard packages,
`spec` normalises and checks a design, `layout`/`render` draw it, `build`
writes the same files an ingest writes so the rest of the system sees an
ordinary sheet.
"""
from . import catalog, templates, spec, layout, render, build  # noqa: F401
