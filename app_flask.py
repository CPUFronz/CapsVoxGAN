import os
import io
import time
import uuid

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Flask, Response

from gan import Generator
from constants import Z_SIZE
from constants import SAVED_GENERATOR

torch.manual_seed(int(time.time()))

app = Flask(__name__)
HTML = """
<!DOCTYPE html>
<html>
<head>
    </style>
    <title>CapsVoxGAN</title>
</head>
<body>
        <img style="max-width:100%;max-height:100vh;margin:-75px;" src="/plot.png" alt="Generated Voxel Model">
        <button style="margin:-35%;" onclick="location.reload()">Generate Image</button>
</body>
</html>
"""


@app.route('/plot.png')
def generate_image(threshold=1.0):
    model = Generator()
    model.load_state_dict(torch.load(SAVED_GENERATOR + '_state_dict', map_location='cpu'))
    
    noise = torch.randn(1, Z_SIZE)
    with torch.no_grad():
        generated_model = model(noise).squeeze().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.voxels(generated_model >= threshold, facecolor='blue', edgecolor='k')

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/')
def show_index():
    full_filename = generate_image()
    return HTML


if __name__ == '__main__':
    app.run(host='0.0.0.0')
