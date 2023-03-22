FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
COPY ./requirement.txt /app/requirement.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirement.txt
RUN pip3 install --no-cache-dir torch==1.13.1 \
    torchvision==0.14.1 \
    torchaudio==0.13.1
COPY ./ /app
CMD ["python","main.py"]