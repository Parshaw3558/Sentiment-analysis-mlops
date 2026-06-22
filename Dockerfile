FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data non-interactively
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/share/nltk_data')"

COPY flask_app/ .

COPY models/vectorizer.pkl ./models/vectorizer.pkl
COPY models/model.pkl ./models/model.pkl

ENV NLTK_DATA=/usr/share/nltk_data

EXPOSE 5000

#local
# CMD ["python", "app.py"]

#Prod
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]

