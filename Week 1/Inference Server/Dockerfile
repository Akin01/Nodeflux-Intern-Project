FROM python:3.10-slim-buster
WORKDIR /app

COPY requirements.txt app/
RUN pip install -r app/requirements.txt

COPY images_aligner.py app/
COPY app.py app/
COPY utils.py app/

ENTRYPOINT [ "python" ]
CMD ["app/app.py" ]

EXPOSE 7001