FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
WORKDIR /code
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN apt install -y libglib2.0-0 libsm6 libxrender1 libxext6
COPY requirements.txt /code/
RUN python -m pip install --upgrade pip
RUN pip install --upgrade -r requirements.txt
COPY . /code/