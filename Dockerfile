
# # Specify the base image
# FROM tensorflow/tensorflow:2.10.0

# # Update the package manager and install a simple module. The RUN command
# # will execute a command on the container and then save a snapshot of the
# # results. The last of these snapshots will be the final image
# RUN apt clean && apt-get update -y && apt-get install -y zip graphviz

# # Install additional Python packages
# RUN pip install --upgrade pip
# RUN pip install jupyter pandas scikit-learn matplotlib ipympl ortools pydot\
#     RISE jupyter_contrib_nbextensions tables tensorflow_probability==0.14.1 \
#     tensorflow-lattice

# #RUN pip install ipympl==0.9.3 rise==5.7.1 jupyter-contrib-nbextensions==0.7.0
# RUN jupyter contrib nbextension install --user
# #RUN jupyter contrib nbextension install --system

# # Make sure the contents of our repo are in /app
# COPY . /app

# # Specify working directory
# WORKDIR /app/notebooks

# # Use CMD to specify the starting command
# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
#      "--ip=0.0.0.0", "--allow-root"]

FROM python:3.11

# Update the package manager and install a simple module. The RUN command
# will execute a command on the container and then save a snapshot of the
# results. The last of these snapshots will be the final image
RUN apt-get update -y && apt-get install -y zip graphviz

# Make sure the contents of our repo are in /app
COPY . /app

# Specify working directory
WORKDIR /app

# Install main packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install Python packages for editing and to support presentation mode
RUN pip install ipympl==0.9.3 rise==5.7.1 jupyter-contrib-nbextensions==0.7.0
RUN jupyter contrib nbextension install --system

# Specify working directory
WORKDIR /app/notebooks

# Use CMD to specify the starting command
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

