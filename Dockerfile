FROM python:3.11.5-slim

#Set working directory
WORKDIR /app

#Copy the rest of the application files
COPY . /app

# Ensure all dependencies are available
RUN python -m pip install --no-cache-dir -r requirements.txt


#Expose the port
EXPOSE 8000