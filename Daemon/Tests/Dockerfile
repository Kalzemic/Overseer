FROM python:3.11-slim

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers

WORKDIR /workspace
COPY stride_training.py .

CMD ["python", "stride_training.py"]