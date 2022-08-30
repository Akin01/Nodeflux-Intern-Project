FROM python:3.10-slim-buster
WORKDIR /app

COPY requirements.txt app/
RUN pip install -r app/requirements.txt
RUN pip install opencv-python-headless

COPY models/ models/
COPY utils.py .
COPY classifier.py .
COPY app.py .

ENTRYPOINT [ "python" ]
CMD ["app.py" ]

EXPOSE 7001