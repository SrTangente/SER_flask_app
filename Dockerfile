# Use an official Python runtime as a parent image
FROM jupyter/tensorflow-notebook

USER root

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD ./app /app

# Download and unzip
RUN wget http://emodb.bilderbar.info/download/download.zip
RUN unzip download.zip -d /app/data

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
RUN pip install jupyterlab

# Install Supervisor
RUN apt-get update && apt-get install -y supervisor
RUN mkdir -p /var/log/supervisor

# Copy Supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Make ports available
EXPOSE 8888 5555

# Run Supervisor
CMD ["/usr/bin/supervisord"]