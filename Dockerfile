FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
ADD sources.list /etc/apt/
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /docker_sample
COPY . /docker_sample

RUN apt-get update && apt-get install -y vim python3 python3-pip

RUN pip3 install --upgrade pip --default-timeout=6000 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install tensorboard==2.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install torch_scatter-2.0.7-cp38-cp38-linux_x86_64.whl


EXPOSE 80

