FROM python:3.12-slim


RUN pip install pipenv


WORKDIR /app


COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system


COPY ["predict.py", "xgb_model.pkl", "pipeline.pkl", "./"]


EXPOSE 9696


ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
