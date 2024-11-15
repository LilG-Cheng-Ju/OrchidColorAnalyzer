FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1

CMD ["python", "app.py"]