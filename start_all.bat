@echo off
echo Arrancando GENIE Learn ecosystem...
start "GENIE Learn" cmd /k "cd /d c:\Users\Paten\Desktop\genie-learn-cursor && streamlit run app.py --server.port 8501"
timeout /t 3
start "EasyVis Demo" cmd /k "cd /d c:\Users\Paten\Desktop\genie-learn-cursor && streamlit run dashboard/demo_completo.py --server.port 8503"
timeout /t 3
start "UBUN.IA Builder" cmd /k "cd /d c:\Users\Paten\Desktop\genie-learn-cursor && streamlit run ubun_builder.py --server.port 8511"
timeout /t 3
echo.
echo ✓ Apps corriendo:
echo   GENIE Learn      → http://localhost:8501
echo   EasyVis Demo     → http://localhost:8503
echo   UBUN.IA Builder  → http://localhost:8511
echo.
pause
