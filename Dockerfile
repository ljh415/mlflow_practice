FROM python:3.9

RUN pip install mlflow==2.1 \
    && pip install cloudpickle==2.2.1 \
    && pip install ipython==8.10.0 \
    && pip install torch==1.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 \
    && pip install torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu113 \
    && pip install torchmetrics==0.11.1