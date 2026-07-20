#!/usr/bin/env python3
"""Compatibility wrapper for the demo-only synthetic HTTP server.

Use demo_stub_http_server.py for new scripts. Real and replay operation use
flight_operator_app.py.
"""
from demo_stub_http_server import main


if __name__ == "__main__":
    main()
