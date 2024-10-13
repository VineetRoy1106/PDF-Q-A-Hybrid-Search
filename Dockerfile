# Step 1: Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Step 4: Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the application code into the container
COPY . .

# Step 6: Set environment variables (if you want defaults)
# You can set default values here, but it's safer to use the .env file.
ENV PINECONE_API_KEY=PINECONE_API_KEY
ENV GROQ_API_KEY=GROQ_API_KEY

# Run the NLTK downloader
RUN python -m nltk.downloader punkt

# Step 7: Expose the port the app runs on (assuming you are using Streamlit or another service running on port 8501)
EXPOSE 8501

# Step 8: Define the command to run your application
CMD ["streamlit", "run", "app.py"]