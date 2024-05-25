FROM python:3.9-alpine

RUN mkdir -p /home/python/paraphraser

WORKDIR /home/python/paraphraser

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 65535

ENTRYPOINT [ "python"]

CMD [ "service.py" ]
