FROM ubuntu:16.04
ADD sources.list /etc/apt/
#RUN apt-get update && apt-get install -y vim && apt-get install -y nginx
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /docker_sample

COPY . /docker_sample
RUN apt-get update && apt-get install -y vim python3.8 python3-pip

RUN pip3 install --upgrade pip --default-timeout=6000

RUN pip3 install -U scipy scikit-learn --default-timeout=6000
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install tensorboard==2.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 80

CMD ["python3", "train_1028.py"]
