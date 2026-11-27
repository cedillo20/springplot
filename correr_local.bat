@echo off
rem Inicia SpringPlot en local (Windows)
setlocal
cd /d "%~dp0"
title SpringPlot local

rem Buscar Python 3.12
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
  python --version 2>nul | findstr /b /c:"Python 3.12" >nul && set "PY=python"
)
if not defined PY (
  echo No encontre Python 3.12.
  echo Instalalo desde https://www.python.org/downloads/ y vuelve a abrir este archivo.
  pause
  exit /b 1
)

rem Crear el entorno e instalar las dependencias si cambió requirements.txt
if not exist ".venv\Scripts\python.exe" (
  echo Creando el entorno de Python ^(solo la primera vez^)...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo No se pudo crear el entorno de Python.
    pause
    exit /b 1
  )
)
fc /b "backend\requirements.txt" ".venv\requirements.instalado" >nul 2>&1
if errorlevel 1 (
  echo Instalando dependencias. La primera vez tarda varios minutos...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r "backend\requirements.txt"
  if errorlevel 1 (
    echo Fallo la instalacion de dependencias. Revisa tu conexion a internet y vuelve a intentar.
    pause
    exit /b 1
  )
  copy /y "backend\requirements.txt" ".venv\requirements.instalado" >nul
)

rem Iniciar el servidor y abrir el navegador
echo.
echo SpringPlot corriendo en http://127.0.0.1:8000
echo Para detenerlo, cierra esta ventana.
echo.
if not defined SPRINGPLOT_SIN_NAVEGADOR start "" /min cmd /c "timeout /t 8 /nobreak >nul & start http://127.0.0.1:8000"
cd backend
"..\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
