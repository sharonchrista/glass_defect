@echo off
REM Glass Defect Project - Folder Structure Setup
REM Run this by double-clicking, or from CMD: setup_folders.bat

echo Creating folder structure at D:\glass_defect ...

mkdir "D:\glass_defect"
mkdir "D:\glass_defect\dataset"
mkdir "D:\glass_defect\pipeline"
mkdir "D:\glass_defect\pipeline\scripts"
mkdir "D:\glass_defect\pipeline\dashboard"
mkdir "D:\glass_defect\pipeline\dashboard\templates"
mkdir "D:\glass_defect\pipeline\docs"
mkdir "D:\glass_defect\work"
mkdir "D:\glass_defect\models"

echo.
echo Done. Structure created:
echo.
echo D:\glass_defect
echo   ^|-- dataset\       (put archive.zip and roboflow.zip here)
echo   ^|-- pipeline\      (all the code goes here)
echo   ^|   ^|-- scripts\
echo   ^|   ^|-- dashboard\
echo   ^|   ^|   ^|-- templates\
echo   ^|   ^|-- docs\
echo   ^|-- work\          (auto-filled by training script, leave empty)
echo   ^|-- models\        (auto-filled by training script, leave empty)
echo.
echo Next: copy the provided files into pipeline\ matching this layout,
echo then confirm your zips are sitting in dataset\
echo.
pause
