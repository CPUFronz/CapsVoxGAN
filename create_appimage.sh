#!/usr/bin/env bash

cd deployment/AppImage/

wget -c "https://raw.githubusercontent.com/TheAssassin/linuxdeploy-plugin-conda/master/linuxdeploy-plugin-conda.sh"
wget -c "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
chmod +x linuxdeploy-x86_64.AppImage linuxdeploy-plugin-conda.sh

cat > capsvoxgan.desktop <<\EOF
[Desktop Entry]
Version=1.0
Name=CapsVoxGAN
Exec=cvg
Terminal=false
Type=Application
Icon=capsvoxgan
Categories=Graphics;Science;Engineering;
StartupNotify=true
EOF

export CONDA_PYTHON_VERSION=3.7.4
export CONDA_CHANNELS=pytorch
export CONDA_PACKAGES="pyqt==5.9.2;matplotlib==3.1.1;pytorch-cpu==1.1.0;torchvision-cpu==0.2.2;tqdm==4.40.0;h5py==2.9.0;tensorboard==2.0.0;future==0.18.2"
./linuxdeploy-plugin-conda.sh --appdir AppDir

# copying the app to AppDir
APP_DIR=AppDir/usr/opt/CapsVoxGAN
mkdir -p $APP_DIR
cp ../../gan.py $APP_DIR/gan.py
cp ../../gui_app.py $APP_DIR/gui_app.py
cp ../../voxeldata.py $APP_DIR/voxeldata.py
cp ../../constants.py $APP_DIR/constants.py
cp ../../generator.pkl $APP_DIR/generator.pkl
cp cvg $APP_DIR/cvg

cd AppDir/usr/bin/
ln -s ../opt/CapsVoxGAN/cvg
chmod +x cvg

cd ../../../
./linuxdeploy-x86_64.AppImage --appdir AppDir -i capsvoxgan.png -d capsvoxgan.desktop --output appimage
mv $(ls CapsVoxGAN*) ../../CapsVoxGAN.AppImage

rm -r AppDir/
rm -r _temp_home/
rm capsvoxgan.desktop
rm linuxdeploy-plugin-conda.sh
rm linuxdeploy-x86_64.AppImage

