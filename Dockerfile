# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=.

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock /app/

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the application
COPY . /app/

# Ensure data directories exist
RUN mkdir -p data/raw data/processed data/faiss_index

# Expose the port the app runs on
EXPOSE 8000

# We'll use a wrapper script to check for index or just run the app
# Note: Rebuilding index requires a MISTRAL_API_KEY, so we can't do it at build time easily.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
