FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
WORKDIR /opt/CapsVoxGAN
ADD gan.py .
ADD app_flask.py .
ADD voxeldata.py .
ADD constants.py .
ADD generator.pkl .
RUN pip install tqdm==4.40.0 flask==1.1.1 tensorboard==2.0.0 matplotlib==3.1.1 future==0.18.2 h5py==2.9.0
EXPOSE 5000
CMD ["python", "/opt/CapsVoxGAN/app_flask.py"]
