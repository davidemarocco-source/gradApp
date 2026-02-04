@echo off
cd /d "%~dp0"
echo Installing/Verifying dependencies...
pip install -r requirements.txt
echo Starting OpenOMR Grading System...
echo Note: This will open in your web browser.
python -m streamlit run app.py --browser.gatherUsageStats=false
pause
