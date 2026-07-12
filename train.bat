@echo off
echo ==========================================
echo Starting Model Training Pipeline
echo ==========================================
.\venv\python.exe -m src.Pipeline.train_pipeline %*
pause
