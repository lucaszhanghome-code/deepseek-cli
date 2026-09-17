@echo off
rem Launcher for Windows: forwards any arguments to the deepseek CLI.
setlocal
python -m deepseek_cli %*
