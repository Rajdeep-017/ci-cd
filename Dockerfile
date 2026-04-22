FROM python:3.9-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model during image build
RUN python app/train.py

# Run a test prediction when the container starts
CMD ["python", "app/predict.py", "5.1", "3.5", "1.4", "0.2"]