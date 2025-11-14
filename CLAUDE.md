# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project named "stackai" that uses modern Python tooling (requires Python 3.12+). The project is in its early stages with minimal functionality.

## Development Setup

This project uses `pyproject.toml` for dependency management. The project appears to be set up for use with `uv` or similar modern Python package managers.

**Install dependencies:**
```bash
uv sync
```

**Run the main script:**
```bash
python main.py
# or
uv run main.py
```

## Project Structure

- `main.py` - Entry point with a simple main() function
- `pyproject.toml` - Project configuration and dependencies

## Architecture Notes

The project currently has a minimal structure with no defined architecture. As the project grows, architecture decisions should be documented here.
