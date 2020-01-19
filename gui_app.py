import time
import uuid
from tempfile import TemporaryDirectory

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.config import Config

from gan import Generator
from constants import Z_SIZE
from constants import SAVED_GENERATOR

Config.set('graphics', 'window_state', 'maximized')
torch.manual_seed(int(time.time()))


class Window(FloatLayout):
    def __init__(self, **kwargs):
        super(Window, self).__init__(**kwargs)

        self.model = torch.load(SAVED_GENERATOR, map_location='cpu')
        self.plots_directory = TemporaryDirectory()

        img = self.generate_image()
        self.current_image = Image(source=img)
        self.add_widget(self.current_image)

        self.submit = Button(text="Generate Image", size_hint=(.25/1.5, .125/2), pos=(20, 20))
        self.submit.bind(on_press=self.pressed)
        self.add_widget(self.submit)

    def pressed(self, instance):
        import time
        print('button', time.time())
        image = self.generate_image()
        self.current_image.source = image

    def generate_image(self, threshold=1.0):
        filename = self.plots_directory.name + '/' + str(uuid.uuid1()) + '.png'

        noise = torch.randn(1, Z_SIZE)
        generated_model = self.model(noise).squeeze().detach().numpy()

        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ax.voxels(generated_model >= threshold, facecolor='blue', edgecolor='k')
        plt.savefig(filename, format='png')

        return filename


class Main(App):
    def build(self):
        return Window()


if __name__ == '__main__':
    Main().run()
