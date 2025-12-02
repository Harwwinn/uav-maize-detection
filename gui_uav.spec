# gui_uav.spec
# -*- mode: python ; coding: utf-8 -*-

import os
from glob import glob
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Ruta del proyecto (carpeta con main.py y este .spec)
project_dir = os.getcwd()


# =======================================
# ARCHIVOS DE DATOS (iconos, qss, ui, modelo, etc.)
# =======================================
datas = []
binaries = []
hiddenimports = [
    "PyQt5.sip",
    "PyQt5.QtWebEngineWidgets",
    "PyQt5.QtWebChannel",
]

# Archivos sueltos en el root del proyecto
single_files_patterns = [
    "*.ico",        # todos los iconos .ico
    "*.png",        # Logo.png y otros png que tengas ahí
    "*.qss",        # style.qss
    "*.ui",         # design.ui, diseño_ventana.ui
    "*.qrc",        # resource.qrc
]

for pattern in single_files_patterns:
    for f in glob(os.path.join(project_dir, pattern)):
        datas.append((f, "."))   # los ponemos en el root del bundle

# Carpetas con recursos SOLO DE LECTURA
folder_mappings = [
    ("models", "models"),   # densenet_201_fold4.pth
    ("icons",  "icons"),    # ./icons/field_map.png
    # fotos_path y diagnosticos_guardados NO las meto aquí
    # porque son carpetas de trabajo/escritura (se crean en runtime)
]

for src_folder, dest_folder in folder_mappings:
    full_src = os.path.join(project_dir, src_folder)
    if os.path.isdir(full_src):
        datas.append((os.path.join(full_src, "*"), dest_folder))

# =======================================
# PYQT5 + QTWEBENGINE (archivos extra)
# =======================================
qtwe_datas, qtwe_bins, qtwe_hidden = collect_all("PyQt5.QtWebEngineWidgets")
datas += qtwe_datas
binaries += qtwe_bins
hiddenimports += qtwe_hidden

# =======================================
# ANALYSIS
# =======================================
a = Analysis(
    ['main.py'],
    pathex=[project_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# =======================================
# EXE (onefile, sin consola)
# =======================================
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GUI_UAV',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # PyQt5 sin consola
    icon=os.path.join(project_dir, 'dashboard-5-48.ico'),  # cambia si quieres otro icono
)
