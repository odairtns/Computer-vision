FROM python:3.9
WORKDIR /app/backend
COPY backend/ .
RUN pip install -r requirements.txt
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]