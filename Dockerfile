FROM python:3.10-slim

#Set working directory
WORKDIR /app

#Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy the rest of the application files
COPY . .

#Expose the port
EXPOSE 8501

#Run the startup script
ENTRYPOINT [ "./startup.sh" ]