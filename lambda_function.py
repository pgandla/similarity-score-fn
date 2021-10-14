import json
import os

import boto3

import nltk
import psycopg2
from nltk import stopwords, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

session = boto3.Session()
s3 = session.resource('s3')
sqs = boto3.client('sqs')

nltk.download('stopwords')
nltk.download('punkt')


stop_words = set(stopwords.words('english'))
MAX_SIZE = 1300
bucket = os.getenv("EXTRACTED_TEXT_BUCKET")
output_queue_url = os.getenv("SIMILARITY_DB_SQS_URL")
reference_source = os.getenv("REF_SOURCE")
db_params = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS")
}


def download_text(document_id):
    return s3.Object(bucket, document_id + "/text.txt").get()['Body'].read().decode('utf-8')


def document_similarity(doc_id1, doc_id2, doc_1, doc_2):
    compare_docs = [doc_1, doc_2]
    vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    x = vectorizer.fit_transform(compare_docs)
    similarity_score = cosine_similarity(x[0], x[1])
    message = {"InputDocId": doc_id1, "CompDocId": doc_id2,
               "Score": similarity_score[0][0] * 100}
    return message


def remove_stopwords(text):
    tot_words = ""
    words = word_tokenize(text)
    for w in words:
        if w not in stop_words:
            tot_words = tot_words + " " + w
    return tot_words


def lambda_handler(event, context):
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
