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
from django.http import JsonResponse









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



def process_video(request):
    if request.method == 'POST':
        video_url = request.POST.get('video_url', None)
        action = request.POST.get('action', None)

        if video_url:
            if action == 'fetch_comments':              
                return fetch_comments(request)
            elif action == 'display_comments':             
                return display_comments(request)
            elif action == 'classify_comments':              
                return classify_comments(request)
    else:        
        return render(request, 'home.html')



# Create your views here.
def home(request):
    return render(request, 'home.html')

def svm(request):
    return render(request, 'svm.html')



def get_youtube_comments(api_key, video_id, max_results=30):
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

# fetch commments and display
def fetch_comments(request):
    if request.method == 'POST':
        video_url = request.POST.get('video_url', None)
        if video_url:
            video_id = extract_video_id(video_url)
            api_key = "AIzaSyCw7KvWAHPK1SHA3tyEM2f2JcWbZ6jcEd0" 
            comments = get_youtube_comments(api_key, video_id)
            return render(request, 'home.html', {'comments': comments})
    else:
        return render(request, 'home.html')

#SVM-Classification
def display_comments(request, video_id=None):
    api_key = "AIzaSyCw7KvWAHPK1SHA3tyEM2f2JcWbZ6jcEd0"
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
        for index, row in comments_df.head(2).iterrows():
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

    
    return render(request, 'svm.html', context)



# Define MultinomialNaiveBayes classification 


def classify_comments(request, video_id= None):
    api_key = "AIzaSyCw7KvWAHPK1SHA3tyEM2f2JcWbZ6jcEd0"
    comments= []
    comments_df = pd.DataFrame(columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
    # comments_df.head(10)
    
    
    if not video_id and request.method == 'POST':
        video_url = request.POST.get('video_url', None)
        if video_url:
            video_id = extract_video_id(video_url)
            return redirect('display_classification_with_id', video_id=video_id)
        
    classification_list = []
    if video_id:
        comments = get_youtube_comments(api_key, video_id)
        
        
                
        comments_df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
        # print(comments_df['text'][0])
        for index, row in comments_df.head(5).iterrows():
            comment = row['text']
            comment_preprocessed = preprocess_text(comment)
        
            # Load the saved models and CountVectorizer
            label_plot = ['toxic', 'obscene', 'insult']
            models = {}

            #Initialize the mnbmodels
            for label in label_plot:
                models[label] = joblib.load(f"nmbmodel_{label}.joblib")

            #    Initialize the vectorizer file
            count_vector = joblib.load("count_vector.joblib")
            comments_vectorized = count_vector.transform([comment_preprocessed]).toarray()
            
            predictions= {}
            probabilities = {}
            for label, model in models.items():
                probability = model.predict_proba(comments_vectorized)
                predictions[label] = np.argmax(probability)
                probabilities[label] = probability[0][1]  # Probability of being in class 1 (toxic, obscene, or insult)
                
            
                
                # prediction, probabilities = model.predict(comments_vectorized)
            
                
            # classification_list.append((row.to_dict(), predictions.tolist()))
            
            # Print predictions for each comment
            print(f"Comment {index + 1}: {comment}")
            for label, prediction in predictions.items():
                print(f"Predicted {label}: {prediction}")
                print(f"Probability {label}: {probabilities[label]}")
            print()
            # classification_list['predictions'] = prediction
            # classification_list['probabilities'] = probabilities    
             
            # classification_list.append((row.to_dict(), predictions, probabilities[label]))\
            classification_list.append({
                'comment_info': row.to_dict(),
                'predictions': predictions,
                'probabilities': probabilities
            })
            # print(probabilities)
    # print(classification_list[0])

    context = {
        'comments': comments, 
        'video_id': video_id,
        
        'classification_list': classification_list,
    
    }
                
                

    return render(request, 'classify_comments.html',  context=context)
    



from django.shortcuts import render
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss, accuracy_score, log_loss, multilabel_confusion_matrix
import joblib


def svm_evaluation(request):
    history = joblib.load('training_history1.joblib')
    
    X_test = history['X_test']
    Y_test = history['Y_test']
    predictions_loaded = history['predictions_loaded']
    
    label_plot = ['toxic', 'obscene', 'insult']

    # Generate classification report
    class_report = classification_report(Y_test, predictions_loaded,target_names=label_plot)

    # Print classification report
    print("Classification Report:\n", class_report)

    # Calculate overall accuracy
    accuracy = accuracy_score(Y_test, predictions_loaded)
    print("Overall Accuracy : {}".format(accuracy*100))

    # Calculate Hamming loss
    hamming_loss_value = hamming_loss(Y_test, predictions_loaded)
    print("Hamming_loss : {}".format(hamming_loss_value*100))



    # Generate the confusion matrix
    confusion_matrices = multilabel_confusion_matrix(Y_test, predictions_loaded)

    confusion_matrices_dict = {}
    for i, label in enumerate(label_plot):
        confusion_matrices_dict[label] = confusion_matrices[i]
        
    context = {
        'class_report': class_report,
        'overall_accuracy': accuracy * 100,
        'hamming_loss': hamming_loss_value * 100,
        'confusion_matrices': confusion_matrices_dict
    }

    return render(request, 'svm_evaluation.html', context)

def display_evaluation(request):
    # Load evaluation data
    evaluation_data = joblib.load("evaluation_data.joblib")

    # Extract necessary information
    true_labels = evaluation_data['true_labels']
    
    predicted_labels = evaluation_data['predicted_labels']
    label_plot = evaluation_data['label_plot']

    # Generate classification report and confusion matrix
    reports = []
    confusion_matrices = {}
    for i, label in enumerate(label_plot):
        report = classification_report(true_labels[:, i], predicted_labels[:, i], output_dict=True)
        matrix = confusion_matrix(true_labels[:, i], predicted_labels[:, i])
        reports.append((label, report))
        confusion_matrices[label] = matrix
    print(confusion_matrices)
    # Calculate evaluation scores
    loss = hamming_loss(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    try:
        logloss = log_loss(true_labels, predicted_labels)
    except:
        logloss = log_loss(true_labels, predicted_labels.toarray())

    # Render the template with the results
    return render(request, 'evaluation_results.html', {
        'reports': reports,
        'confusion_matrices': confusion_matrices,
        'hamming_loss': loss * 100,
        'accuracy': accuracy * 100,
        'log_loss': logloss,
    })
