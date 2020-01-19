#!/usr/bin/env bash

cd deployment/AppImage/

wget -c "https://raw.githubusercontent.com/TheAssassin/linuxdeploy-plugin-conda/master/linuxdeploy-plugin-conda.sh"
wget -c "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
chmod +x linuxdeploy-x86_64.AppImage linuxdeploy-plugin-conda.sh

cat > capsvoxgan.desktop <<\EOF
[Desktop Entry]
Version=1.0
Name=CapsVoxGAN
Exec=gui_app.py
Terminal=false
Type=Application
Icon=capsvoxgan
Categories=Graphics;Science;Engineering;
StartupNotify=true
EOF

export CONDA_CHANNELS=pytorch
export CONDA_PACKAGES="pyqt==5.9.2;matplotlib==3.1.1;pytorch-cpu==1.1.0;torchvision-cpu==0.2.2"
./linuxdeploy-plugin-conda.sh --appdir AppDir

# copy files to AppDir
APP_DIR=AppDir/usr/conda/opt/CapsVoxGAN
mkdir -p $APP_DIR
cp ../../gan.py $APP_DIR/gan.py
cp ../../gui_app.py $APP_DIR/gui_app.py
cp ../../voxeldata.py $APP_DIR/voxeldata.py
cp ../../constants.py $APP_DIR/constants.py
cp ../../generator.pkl $APP_DIR/generator.pkl

cd AppDir/usr/bin/
ln -s ../conda/opt/CapsVoxGAN/gui_app.py
chmod +x gui_app.py

cd ../../../
#./linuxdeploy-x86_64.AppImage --appdir AppDir -i capsvoxgan.png -d capsvoxgan.desktop --output appimage

