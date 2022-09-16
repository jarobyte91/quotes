FROM requirements:latest
COPY lib.py app.py wsgi.py runtime.txt Procfile ./
RUN pip install gunicorn
USER quotes
CMD gunicorn --bind 0.0.0.0:$PORT wsgi 
