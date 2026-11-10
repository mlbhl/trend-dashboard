"""Dash callbacks. register_callbacks(app) wires the sidebar, results and optimization callbacks."""

from . import optimization_callbacks, results_callbacks, sidebar_callbacks


def register_callbacks(app):
    """Register all callbacks for the Dash app."""
    sidebar_callbacks.register(app)
    results_callbacks.register(app)
    optimization_callbacks.register(app)
