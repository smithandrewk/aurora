FROM ubuntu:latest
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NOWARNINGS="yes"
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils
COPY aurora /aurora
RUN apt-get update -y && \
	apt-get update -y && \
	apt-get upgrade -y && \
	apt-get autoremove -y && \
	apt-get install -y apt-utils \
			   python3 \
			   python3-pip \
			   tmux
RUN cd aurora && pip3 install -r requirements.txt