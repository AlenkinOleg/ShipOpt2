FROM ubuntu:16.04

RUN apt-get update &&\
    apt-get install -y python python-pip &&\
    apt-get install -y python-tk &&\
    pip install numpy matplotlib scipy &&\
    pip install GPy &&\
    pip install GPyOpt&&\
    apt-get clean

RUN mkdir /output/; ls output

COPY /src/* /opt/
