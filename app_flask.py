import os
import time
import uuid
from tempfile import TemporaryDirectory

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from flask import Flask, render_template

from gan import Generator
from constants import Z_SIZE
from constants import SAVED_GENERATOR

torch.manual_seed(int(time.time()))

PLOTS_FOLDER = TemporaryDirectory()
app = Flask(__name__, static_url_path='', static_folder=PLOTS_FOLDER.name, template_folder='deployment/Flask')
app.config['UPLOAD_FOLDER'] = ''


def generate_image(threshold=1.0):
    model = torch.load(SAVED_GENERATOR, map_location='cpu')

    filename = PLOTS_FOLDER.name + '/' + str(uuid.uuid1()) + '.png'

    noise = torch.randn(1, Z_SIZE)
    generated_model = model(noise).squeeze().detach().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.voxels(generated_model >= threshold, facecolor='blue', edgecolor='k')
    plt.savefig(filename, format='png')

    return filename

@app.route('/')
@app.route('/index')
def show_index():
    full_filename = generate_image(threshold=0.9)
    return render_template("index.html", user_image = full_filename)