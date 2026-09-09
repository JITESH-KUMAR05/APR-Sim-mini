@echo off
title PaintPilot - Automated Painting Rover (APR) Console
echo ===================================================================
echo   PAINTPILOT - WALL-TO-3D CAD & SURFACE SEGMENTATION CONSOLE
echo   Automated Painting Rover (APR) - MSME Grant Ref: INC25ETS084848
echo   WBS 1.2: 3D Geometry & Coverage Planning Subsystem
echo ===================================================================
echo.
echo Launching server at http://127.0.0.1:5000 ...
start "" "http://127.0.0.1:5000"
python app.py
pause
