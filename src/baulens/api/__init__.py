"""HTTP interface for inspection workflows: ``create_app(config)`` builds the FastAPI app."""

from baulens.api.app import create_app

__all__ = ["create_app"]
