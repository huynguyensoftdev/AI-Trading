FROM python:3.10-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY pyproject.toml /app/
RUN apt-get update && apt-get install -y build-essential git
RUN pip install --upgrade pip
RUN pip install -e ".[dev]"
COPY . /app
CMD ["bash"]
