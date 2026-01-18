FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

# 🔥 FORCE clean OpenCV install
RUN pip uninstall -y opencv-python || true \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
