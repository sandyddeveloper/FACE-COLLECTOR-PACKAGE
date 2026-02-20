# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and MTCNN
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Install the package locally
RUN pip install -e .

# Create the output directory
RUN mkdir -p output

# Default command to run the package
# Use ENV variables or Command line args to override defaults
ENTRYPOINT ["face-collector"]
CMD ["--stream-url", "http://192.168.68.103:8080/video", "--output-dir", "output"]
