FROM nvcr.io/nvidia/tensorflow:20.08-tf1-py3
RUN sed -i "1iforce_color_prompt=true" "$HOME/.bashrc"
COPY . /LearnedISP
RUN apt update \
&& apt install -y htop vim curl zip
RUN cd /LearnedISP \
&& pip install --upgrade pip \
&& pip install -r requirements.txt
WORKDIR /LearnedISP