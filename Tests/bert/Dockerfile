FROM python:3.11-slim

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers

WORKDIR /workspace
COPY bert_training_basic.py .

CMD ["python", "bert_training_basic.py"]