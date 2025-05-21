# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

# Copy the application code
COPY . .


# Expose the port Chainlit runs on
EXPOSE 8000

# Set environment variables for Chainlit and Azure authentication
ENV CHAINLIT_HOST=0.0.0.0
ENV CHAINLIT_PORT=8000
ENV CHAINLIT_HIDE_BRANDING=true
ENV WEBSITES_PORT=8000

# Command to run the app
CMD ["chainlit", "run", "apex.py", "--port", "8000"]
