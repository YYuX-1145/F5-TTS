@echo off
set PYTHONPATH=%CD%\src
set PATH=%PATH%;%CD%\ffmpeg\bin\;%CD%\py310\;%CD%\py310\Scripts
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\download\huggingface
py310\python.exe -m pip uninstall -y accelerate
py310\python.exe -m pip install accelerate==1.5.1
pause