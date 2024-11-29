FROM python:3.11.5-slim

#Set working directory
WORKDIR /app


# #Copy site-packages path
# COPY code_env/Lib/site-packages /usr/local/lib/python3.11/site-packages


#Copy the rest of the application files
COPY . /app

# Ensure all dependencies are available
RUN python -m pip install --no-cache-dir -r requirements.txt


#Expose the port
EXPOSE 8501

RUN chmod +x startup.sh

#Run the startup script
ENTRYPOINT [ "./startup.sh" ]