FROM python:3.11.5-slim

#Set working directory
WORKDIR /app

#Copy the rest of the application files
COPY . /app

# Ensure all dependencies are available
RUN python -m pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "server:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]