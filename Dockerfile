FROM python:3.10-slim-buster
RUN pip install --no-cache-dir flask catboost pandas scikit-learn
RUN mkdir /app

COPY app.py /app/app.py
COPY model.py /app/model.py
COPY eda.py /app/eda.py

WORKDIR /app
EXPOSE 5000
#ENTRYPOINT ["python"]
#CMD ["python", "eda.py"]
#CMD ["python", "model.py"]
RUN python eda.py
RUN python model.py
#COPY trained_model.cbm /app
CMD ["python", "app.py"]
