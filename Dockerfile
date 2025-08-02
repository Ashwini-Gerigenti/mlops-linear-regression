# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
# (Optional) Commented out because models/ is not needed
# COPY models/ models/
COPY . .

# Run prediction
COPY run_all.sh .
RUN chmod +x run_all.sh

CMD ["./run_all.sh"]
