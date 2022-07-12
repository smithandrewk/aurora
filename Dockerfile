FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

ENV DEBCONF_NOWARNINGS="yes"

RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils

RUN apt-get update -y && \
	apt-get update -y && \
	apt-get upgrade -y && \
	apt-get autoremove -y && \
	apt-get install -y apt-utils \
			   python3 \
			   python3-pip \
			   zip

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY . /app

RUN make createDB

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]
