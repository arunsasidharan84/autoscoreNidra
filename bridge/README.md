# Rust Bridge (`bridge/`)

This directory will contain the native Rust code that serves as the bridge between the Flutter frontend and the Python backend scoring engine.

## Responsibilities

- **Backend Process Execution**: Spawns the compiled PyInstaller scoring binary (`autoscore-backend`) as a subprocess.
- **Log Streaming**: Captures `stdout`/`stderr` from the backend process and streams it to the Flutter UI asynchronously.
- **Task Management**: Manages execution state and supports cancellation of active jobs.
