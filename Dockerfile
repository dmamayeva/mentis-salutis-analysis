FROM python:3.9-slim

RUN pip install pipenv
WORKDIR /app
COPY ["predict", "./"]

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system
COPY ["app.py", "xgb_model.bin", "pipeline.bin", "./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "app:app"]