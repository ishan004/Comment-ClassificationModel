from django.shortcuts import render, redirect
from googleapiclient.discovery import build
import pandas as pd
from urllib.parse import urlparse, parse_qs
from django.http import JsonResponse, HttpResponseBadRequest
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import joblib
import numpy as np









lemmatizer = WordNetLemmatizer()
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('wordnet')

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token.lower() for token in tokens if token.isalpha()  and token.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(lemmatizer.lemmatize(token,pos="v")) for token in filtered_tokens]
    return " ".join(stemmed_tokens)






# Create your views here.
def home(request):
    return render(request, 'home.html')




def get_youtube_comments(api_key, video_id, max_results=100):
    youtube = build("youtube", "v3", developerKey=api_key)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )
    response = request.execute()

    comments = []

    for item in response.get('items', []):
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append({
            'author': comment['authorDisplayName'],
            'published_at': comment['publishedAt'],
            'updated_at': comment['updatedAt'],
            'like_count': comment['likeCount'],
            'text': comment['textDisplay']
        })
    df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
    # print(df['text'][0])
    return comments

def extract_video_id(video_url):
    parsed_url = urlparse(video_url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get('v', [None])[0]
    return video_id

def display_comments(request, video_id=None):
    api_key = "AIzaSyBTNs4-zHsgrBiwN4nUN8FhEaINLv9Fy58"
    comments= []
    comments_df = pd.DataFrame(columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
    # comments_df.head(10)
    
    
    if not video_id and request.method == 'POST':
        video_url = request.POST.get('video_url', None)
        if video_url:
            video_id = extract_video_id(video_url)
            return redirect('display_comments_with_id', video_id=video_id)
    prediction_list = []
    
    if video_id:
        comments = get_youtube_comments(api_key, video_id)
        
        
                
        comments_df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
        # print(comments_df['text'][0])
        for index, row in comments_df.head(10).iterrows():
            comment = row['text']
            comment_preprocessed = preprocess_text(comment)
        
            load_model = joblib.load("svm_model.joblib")
            count_vector = joblib.load('count_vector.joblib')

            comments_vectorized = count_vector.transform([comment_preprocessed]).toarray()
            # print(comments_vectorized)
        # print(comments_vectorized)
        
        
        # print(load_model)
        
            predictions = load_model.predict(comments_vectorized)
            # print(predictions)
            
            prediction_list.append((row.to_dict(), predictions.tolist()))
        # print(prediction_list)    
            
            print(f"Author: {row['author']}")
            label_plot = ['toxic', 'obscene', 'insult']
            for i, label in enumerate(label_plot):
                print(f"{label}: {predictions[0, i]}")
            print("\n")


    context = {
        'comments': comments, 
        'video_id': video_id,
        'label_plot': label_plot, 
        'prediction_list': prediction_list,
        }

    
    return render(request, 'home.html', context)





    



