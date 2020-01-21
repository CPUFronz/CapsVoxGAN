#!/usr/bin/env python3

import os
import sys
import time
import uuid
from tempfile import TemporaryDirectory

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication

from gan import Generator
from constants import Z_SIZE
from constants import SAVED_GENERATOR

torch.manual_seed(int(time.time()))

# use absolute path of model
PREFIX = os.path.dirname(os.path.realpath(__file__)) + '/'
SAVED_GENERATOR = PREFIX + SAVED_GENERATOR


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.resize(720, 700)
        self.setWindowTitle('CapsVoxGAN')
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)

        self.model = torch.load(SAVED_GENERATOR, map_location='cpu')
        self.plots_directory = TemporaryDirectory()

        self.central_widget = QtWidgets.QWidget(self)

        self.label = QtWidgets.QLabel(self.central_widget)
        self.label.setGeometry(QtCore.QRect(10, 10, 700, 700))
        self.label.setScaledContents(True)
        self.label.setPixmap(QtGui.QPixmap(self.generate_image()))

        self.button = QtWidgets.QPushButton(self.central_widget)
        self.button.setGeometry(QtCore.QRect(320, 630, 120, 25))
        self.button.setText('Generate Image')
        self.button.clicked.connect(self.on_click)

        self.setCentralWidget(self.central_widget)

    def on_click(self):
        image = self.generate_image()
        self.label.setPixmap(QtGui.QPixmap(image))

    def generate_image(self, threshold=1.0):
        plot = self.plots_directory.name + '/' + str(uuid.uuid1()) + '.png'

        noise = torch.randn(1, Z_SIZE)
        with torch.no_grad():
            generated_model = self.model(noise).squeeze().numpy()

        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ax.voxels(generated_model >= threshold, facecolor='blue', edgecolor='k')
        plt.savefig(plot, format='png')
        plt.close()
    
        return plot


if __name__ == '__main__':
    qt_app = QApplication(sys.argv)
    CapsVoxGAN = App()
    CapsVoxGAN.show()
    sys.exit(qt_app.exec_())
