import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:

    # Created the list for all the allowed locations
    allowed_locations = {
        "Albuquerque, New Mexico",
        "Carlsbad, California",
        "Chula Vista, California",
        "Colorado Springs, Colorado",
        "Denver, Colorado",
        "El Cajon, California",
        "El Paso, Texas",
        "Escondido, California",
        "Fresno, California",
        "La Mesa, California",
        "Las Vegas, Nevada",
        "Los Angeles, California",
        "Oceanside, California",
        "Phoenix, Arizona",
        "Sacramento, California",
        "Salt Lake City, Utah",
        "San Diego, California",
        "Tucson, Arizona"
    }

    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string

            # We Parse query parameters here
            query_params = parse_qs(environ["QUERY_STRING"])
            location = query_params.get("location", [None])[0] # location
            start_date = query_params.get("start_date", [None])[0] # start date
            end_date = query_params.get("end_date", [None])[0] # end date

            # Filter reviews based on query parameters
            filtered_reviews = reviews
            # Filering reviews after fetching all of them based on location
            if location:
                filtered_reviews = [
                    review for review in filtered_reviews 
                    if review["Location"].lower() == location.lower()
                ]
            # Then filtering based on the start date
            if start_date:
                start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
                filtered_reviews = [
                    review for review in filtered_reviews 
                    if datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") >= start_date_dt
                ]
            # Then filtering based on end date
            if end_date:
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                filtered_reviews = [
                    review for review in filtered_reviews 
                    if datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") <= end_date_dt
                ]
            
            # Analyze sentiment for each review
            result = []
            for review in filtered_reviews:
                sentiment = self.analyze_sentiment(review["ReviewBody"])
                result.append({
                    "ReviewBody": review["ReviewBody"],
                    "ReviewId": review["ReviewId"],
                    "Location": review["Location"],
                    "Timestamp": review["Timestamp"],
                    "sentiment": sentiment
                })

            # Convert the result to a JSON byte string
            response_body = json.dumps(result, indent=2).encode("utf-8")

            # Setting the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                # Get the length of the POST data
                content_length = int(environ.get("CONTENT_LENGTH", 0))

                # Read the POST data
                post_data = environ["wsgi.input"].read(content_length).decode("utf-8")
                try:
                    post_data = json.loads(post_data)
                except json.JSONDecodeError:
                    # There can be scenarios where the data is not as JSOn then 
                    # We need to parse it using the URL Form encoded
                    post = ""
                    post_data = parse_qs(post_data)
                    post_data = {k: v[0] for k, v in post_data.items()} 
                
                #post_data = json.loads(post_data.decode("utf-8"))

                # Generate a new UUID for the review
                review_id = str(uuid.uuid4())

                # Add a timestamp for the current time
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                 # Validate location parameter
                location = post_data["Location"]
                if location and location not in self.allowed_locations:
                    error_message = json.dumps({
                        "error": f"Invalid location: '{location}'. Allowed locations are: {', '.join(self.allowed_locations)}"
                    }).encode("utf-8")
                    start_response("400 Bad Request", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(error_message)))
                    ])
                    return [error_message]

                # Create a new review record
                new_review = {
                    "ReviewId": review_id,
                    "ReviewBody": post_data["ReviewBody"],
                    "Location": post_data["Location"],
                    "Timestamp": timestamp
                }

                # Append the new review to the reviews list
                reviews.append(new_review)

                # Respond with the newly created review (including sentiment analysis)
                sentiment = self.analyze_sentiment(new_review["ReviewBody"])
                response_body = json.dumps({
                    "ReviewId": new_review["ReviewId"],
                    "ReviewBody": new_review["ReviewBody"],
                    "Location": new_review["Location"],
                    "Timestamp": new_review["Timestamp"],
                    "sentiment": sentiment
                }, indent=2).encode("utf-8")

                # Set the appropriate response headers
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                
                return [response_body]

            except Exception as e:
                error_message = json.dumps({"error": str(e)}).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(error_message)))
                ])
                return [error_message]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()