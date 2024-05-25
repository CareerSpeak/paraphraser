FROM python:3.9-alpine

RUN mkdir -p /home/python/paraphraser

WORKDIR /home/python/paraphraser

ENV HF_HUB_DISABLE_SYMLINKS_WARNING true
ENV HUGGINGFACE_HUB_CACHE $PWD/~/.cache/huggingface
ENV HF_DATASETS_CACHE $PWD/~/.cache/huggingface

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 65535

ENTRYPOINT [ "python"]

CMD [ "service.py" ]
