@echo off
setlocal

cd /d "%~dp0"

echo.
echo [CHECK] Cerco Python (py)...
py -3 --version >nul 2>nul
if errorlevel 1 goto TRY_PYTHON
set "PY=py -3"
goto HAVE_PY

:TRY_PYTHON
echo [CHECK] Cerco Python (python)...
python --version >nul 2>nul
if errorlevel 1 goto NO_PYTHON
set "PY=python"
goto HAVE_PY

:NO_PYTHON
echo.
echo Python non trovato.
echo 1) Installa Python da https://www.python.org/downloads/windows/
echo 2) Durante l'installazione spunta Add Python to PATH
echo.
pause
exit /b 1

:HAVE_PY
echo.
echo [OK] Uso: %PY%
%PY% --version

echo.
echo [CHECK] pip...
%PY% -m pip --version >nul 2>nul
if errorlevel 1 goto FIX_PIP
goto INSTALL_REQS

:FIX_PIP
echo pip non trovato: provo ensurepip...
%PY% -m ensurepip --upgrade
if errorlevel 1 goto PIP_FAIL

:INSTALL_REQS
echo.
echo [1/2] Installo dipendenze...
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto REQS_FAIL

echo.
echo [2/2] Avvio GUI...
set "PYTHONPATH=%CD%\src"
%PY% scripts\gui.py
if errorlevel 1 goto GUI_FAIL

echo.
echo [OK] Chiusura normale.
pause
exit /b 0

:PIP_FAIL
echo.
echo ERRORE: impossibile installare pip.
pause
exit /b 1

:REQS_FAIL
echo.
echo ERRORE durante l'installazione delle dipendenze.
pause
exit /b 1

:GUI_FAIL
echo.
echo ERRORE: la GUI e' terminata con errore.
pause
exit /b 1
