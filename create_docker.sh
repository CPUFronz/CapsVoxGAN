#!/usr/bin/env bash

docker build -t capsvoxgan -f deployment/Flask/Dockerfile .
docker run --name CapsVoxGAN -p 5000:5000 capsvoxgan

