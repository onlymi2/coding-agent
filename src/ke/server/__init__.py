"""Thin local HTTP server for the ke agent runtime."""

from ke.server.app import create_app

__all__ = ["create_app"]
