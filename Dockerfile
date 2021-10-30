FROM continuumio/miniconda3
RUN conda install -c conda-forge jupyter seaborn jax pytest -y
RUN pip install -Uq tfp-nightly[jax]

RUN git clone --depth 1 https://github.com/pymc-devs/pymc /root/pymc
RUN pip install -e /root/pymc

COPY . /root/pathfinder 
RUN pip install -e /root/pathfinder

ENTRYPOINT jupyter notebook --allow-root --notebook-dir=/root/pathfinder/notebooks --no-browser --ip=0.0.0.0
