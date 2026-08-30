$ErrorActionPreference = "Stop"
py -3.10 -m uvicorn api:app --app-dir "$PSScriptRoot" --reload --port 8000
