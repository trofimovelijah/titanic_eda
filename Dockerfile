FROM python:3.10-slim-buster
RUN pip install --no-cache-dir flask catboost pandas scikit-learn
RUN mkdir /app

COPY app.py /app/app.py
COPY model.py /app/model.py
COPY eda.py /app/eda.py

WORKDIR /app
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["eda.py", "model.py", "app.py"]
