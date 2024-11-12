# Required libraries
import yt_dlp
import os
import re
import json
from pydub import AudioSegment
import openai
from openai import OpenAI

import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, opinion_lexicon
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import requests
from io import StringIO
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Create network plot using corrplot
from tqdm import tqdm
import time

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('opinion_lexicon')

# Initialize OpenAI API client
api_key = "" 
client = openai.OpenAI(api_key=api_key)


# List of YouTube video URLs
urls = ['https://www.youtube.com/watch?v=iSbAYnOMOaU'] #['https://www.youtube.com/watch?v=TYMMKftODY4'] #['https://www.youtube.com/watch?v=ZPoHZj2sGJ0'] #['https://www.youtube.com/watch?v=UZmtShh_0A8'] #['https://www.youtube.com/watch?v=IonVNbhEXGE'] # ["https://www.youtube.com/watch?v=Ll99vicSe40"] #['https://www.youtube.com/watch?v=YSajWy460zE'] #['https://www.youtube.com/shorts/0NqJq6ZhhfU'] # ["https://youtube.com/shorts/fgMrDUCPN9A?si=4-7G82x-cA7BKLvV"] #["https://youtu.be/xiyYN2_JKPc?si=rgSwRX4qFeCvsGmi"] #["https://www.youtube.com/shorts/njgruIvF73A"] #["https://youtube.com/shorts/wwdPTSbUHKo?si=YRitosnr77rrseTf"]#["https://youtube.com/shorts/jcNzoONhrmE?si=ZuxuOLep8pW63EFK"] #["https://www.youtube.com/watch?v=1ejfAkzjEhk", "https://www.youtube.com/shorts/3qPldGG8hdE"] #["https://www.youtube.com/shorts/vlbjwg4c8QE", "https://www.youtube.com/shorts/UBDfQMfHvAY"]

# Set up yt-dlp options
ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Select the best video+audio format available in MP4
    'outtmpl': 'downloaded_video.%(ext)s',  # Output file name template: '%(title)s.%(ext)s'
}

# Function to get video information
def get_video_info(url):
    try:
        with yt_dlp.YoutubeDL() as ydl:
            info = ydl.extract_info(url, download=False)
            return info
    except Exception as e:
        print(f"Error retrieving video info: {e}")
        return None

# Create a folder-friendly title
def folder_friendly_title(title):
    return re.sub(r'\W+', '_', title.lower())

# Function to process a single video
def process_video(url):
    info = get_video_info(url)
    if info:
        # Extract video information
        title = info.get('title', 'N/A')
        uploader = info.get('uploader', 'N/A')
        upload_date = info.get('upload_date', 'N/A')
        description = info.get('description', 'N/A')
        duration = info.get('duration', 'N/A')

        # Print video information
        print(f"Processing: {title}")
        # Create a new folder for the video
        folder_name = folder_friendly_title(title)
        folder_name = os.path.join('public', 'packages', folder_name)
        os.makedirs(folder_name, exist_ok=True)

        # Save video information in JSON
        video_info = {
            "title": title,
            "uploader": uploader,
            "upload_date": upload_date,
            "description": description,
            "duration": duration
        }

        with open(os.path.join(folder_name, 'video_info.json'), 'w') as f:
            json.dump(video_info, f, ensure_ascii=False, indent=4)

        # Download the video
        # ydl_opts['outtmpl'] = os.path.join(folder_name, '%(title)s.%(ext)s')  # Update output template
        ydl_opts['outtmpl'] = os.path.join(folder_name, 'downloaded_video.%(ext)s')

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.download([url])
                print("Download successful.")
        except Exception as e:
            print(f"Error downloading video: {e}")

    #     # Retrieve the downloaded file name and extract audio
    #     try:
    #         downloaded_files = [f for f in os.listdir(folder_name) if f.endswith('.mp4')]
    #         if downloaded_files:
    #             downloaded_file = downloaded_files[0]
    #             downloaded_file_path = os.path.join(folder_name, downloaded_file) # 
    #             print(f"Downloaded file: {downloaded_file}")
                
    #             # Extract the audio and save it as an MP3 file
    #             video = AudioSegment.from_file(downloaded_file_path, format="mp4")
    #             audio_file_path = os.path.join(folder_name, 'final_audio.mp3')
    #             video.export(audio_file_path, format="mp3")
    #             print(f"Audio extracted: {audio_file_path}")
    #         else:
    #             print("No downloaded MP4 file found.")
    #     except Exception as e:
    #         print(f"Error processing video/audio: {e}")
    #     return folder_name, audio_file_path
    # else:
    #     print("Failed to retrieve video information.")
    #     return None, None

            # Retrieve the downloaded file name and extract audio

        try:
            # folder_name = os.path.join('public', 'package', folder_name)
            downloaded_file_path = os.path.join(folder_name, 'downloaded_video.mp4')
            if os.path.exists(downloaded_file_path):
                print(f"Downloaded file: {downloaded_file_path}")

                # Extract the audio and save it as an MP3 file
                video = AudioSegment.from_file(downloaded_file_path, format="mp4")
                audio_file_path = os.path.join(folder_name, 'final_audio.mp3')
                video.export(audio_file_path, format="mp3")
                print(f"Audio extracted: {audio_file_path}")
            else:
                print("No downloaded MP4 file found.")
        except Exception as e:
            print(f"Error processing video/audio: {e}")
        return folder_name, audio_file_path
    else:
        print("Failed to retrieve video information.")
        return None, None


def transcribe_audio(audio_file_path):
    """Transcribe an audio file using OpenAI's Whisper API."""
    transcriptions = {}
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
            transcriptions['transcription_text'] = transcription.text
            transcriptions['segments'] = transcription.segments
            print("Transcription successful!")
    except Exception as e:
        print(f"Error transcribing audio: {e}")

    return transcriptions

# def save_transcription_to_json(transcription_data, output_file_path):
#     """Save transcription data to a JSON file."""
#     with open(output_file_path, 'w') as json_file:
#         json.dump(transcription_data, json_file, ensure_ascii=False, indent=4)
#     print(f'Transcription saved to: {output_file_path}')

def save_transcription_to_json(transcription_data, output_file_path):
    """Save transcription data to a JSON file."""
    with open(output_file_path, 'w') as json_file:
        json.dump(
            transcription_data, 
            json_file, 
            ensure_ascii=False, 
            indent=4, 
            default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)
        )
    print(f'Transcription saved to: {output_file_path}')


def convert_json_to_dataframe(folder_name, json_file_path):
    """Convert JSON file into a pandas DataFrame with specified columns."""
    try:
        # Load the JSON data from the file
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Check if 'segments' key exists in the JSON data
        if 'segments' not in data:
            print("No 'segments' found in the JSON data.")
            return None

        # Extract the 'segments' data
        segments = data['segments']

        # Create a list of dictionaries with the required columns
        segments_data = [
            {
                'segment_id': segment.get('id'),
                'segment_avg_logprob': segment.get('avg_logprob'),
                'segment_compression_ratio': segment.get('compression_ratio'),
                'segment_no_speech_prob': segment.get('no_speech_prob'),
                'seek': segment.get('seek'),
                'start': segment.get('start'),
                'end': segment.get('end'),                
                'temperature': segment.get('temperature'),
                'text': segment.get('text'),
                'tokens': segment.get('tokens')
            }
            for segment in segments
        ]

        # Create a DataFrame from the extracted data
        df_segments = pd.DataFrame(segments_data)
        return df_segments

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



# import corrplot
# from corrr import plot_network




def merge_rows(df, num_rows_to_merge):
    """
    Merge rows in a DataFrame into groups defined by num_rows_to_merge.
    Selects minimum 'start' and maximum 'end' values for merged segments.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to merge.
    num_rows_to_merge (int): The number of rows to merge (either 2 or 3).
    
    Returns:
    pd.DataFrame: A new DataFrame with merged rows.
    """
    if num_rows_to_merge not in [2, 3]:
        raise ValueError("num_rows_to_merge must be either 2 or 3")

    merged_data = []

    for i in range(0, len(df), num_rows_to_merge):
        chunk = df.iloc[i:i + num_rows_to_merge]
        
        merged_row = {}
        
        # Special handling for 'start' and 'end' columns
        merged_row['start'] = chunk['start'].min()
        merged_row['end'] = chunk['end'].max()
        
        for column in df.columns:
            if column == 'text':
                # Concatenate text values with space
                merged_row[column] = ' '.join(chunk[column].astype(str))
            elif column not in ['start', 'end']:
                # Convert to lists for other columns
                merged_row[column] = [chunk[column].tolist()]
        
        merged_data.append(merged_row)
    
    return pd.DataFrame(merged_data)

# Create Time Series DataFrame
def create_time_series(df):
    required_columns = ['start', 'end', #'afinn_positive', 'afinn_negative', 'afinn_net',
                        'nltk_opinion_lexicon_positive', 'nltk_opinion_lexicon_negative',
                        'nltk_opinion_lexicon_net', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
    
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return None

    time_series_data = []

    for _, row in df.iterrows():
        start = row['start']
        end = row['end']
        duration = int(end - start)

        times = np.linspace(start, end, num=duration + 1)
        
        metrics = {col: row[col] for col in required_columns[2:]}
        
        for time in times:
            time_series_data.append({'time': time, **metrics})

    return pd.DataFrame(time_series_data)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(df, output_path='metrics_plot.png'):
    plt.figure(figsize=(14, 8))
    metrics_columns = [
        'nltk_opinion_lexicon_positive', 'nltk_opinion_lexicon_negative', 'nltk_opinion_lexicon_net',
        'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound'
    ]

    for metric in metrics_columns:
        sns.lineplot(data=df, x='time', y=metric, label=metric)

    plt.title('Metrics Over Time', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.legend(title='Metrics', loc='upper right')
    plt.grid()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # If you still want to display the plot, keep this line
    # Otherwise, you can remove it
    # plt.show()

    # Close the figure to free up memory
    plt.close()

# # Plot Metrics
# def plot_metrics(df):
#     plt.figure(figsize=(14, 8))
#     metrics_columns = [
#         'nltk_opinion_lexicon_positive', 'nltk_opinion_lexicon_negative', 'nltk_opinion_lexicon_net',
#         'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound'
#     ]

#     for metric in metrics_columns:
#         sns.lineplot(data=df, x='time', y=metric, label=metric)

#     plt.title('Metrics Over Time', fontsize=16)
#     plt.xlabel('Time (seconds)', fontsize=12)
#     plt.ylabel('Metric Value', fontsize=12)
#     plt.legend(title='Metrics', loc='upper right')
#     plt.grid()
#     plt.show()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to lemmatize words
def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]

# Function to load Hu and Liu lexicon
def load_huliu():
    pos_url = "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/positive-words.txt"
    neg_url = "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/negative-words.txt"
    try:
        pos_words = pd.read_csv(pos_url, header=None, names=['word'], skiprows=35, encoding='utf-8')
        neg_words = pd.read_csv(neg_url, header=None, names=['word'], skiprows=35, encoding='utf-8')
    except UnicodeDecodeError:
        pos_words = pd.read_csv(pos_url, header=None, names=['word'], skiprows=35, encoding='latin-1')
        neg_words = pd.read_csv(neg_url, header=None, names=['word'], skiprows=35, encoding='latin-1')
    pos_words['sentiment'] = 'positive'
    neg_words['sentiment'] = 'negative'
    huliu = pd.concat([pos_words, neg_words])
    huliu['dict'] = 'huliu'
    return huliu

# Function to load Bing lexicon
def load_bing():
    url = "https://raw.githubusercontent.com/dinbav/LeXmo/master/R/sysdata.rda"
    response = requests.get(url)
    content = response.content.decode('latin-1')
    bing = pd.read_csv(StringIO(content), sep='\t', on_bad_lines='skip')
    bing = bing[['word', 'sentiment']]
    bing['dict'] = 'bing'
    return bing

# Function to load NRC lexicon (skipped due to 404 error)
def load_nrc():
    print("NRC lexicon could not be loaded. Skipping...")
    return pd.DataFrame()

# Function to load AFINN lexicon
def load_afinn():
    url = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-en-165.txt"
    afinn = pd.read_csv(url, sep='\t', names=['word', 'score'])
    afinn['sentiment'] = np.where(afinn['score'] < 0, 'negative', 'positive')
    afinn = afinn.drop('score', axis=1)
    afinn['dict'] = 'afinn'
    return afinn

# Function to safely load lexicons
def safe_load(load_func, name):
    try:
        return load_func()
    except Exception as e:
        print(f"Error loading {name} lexicon: {str(e)}")
        return pd.DataFrame()

# Load datasets
huliu = safe_load('path_to_huliu_data', "Hu and Liu")
bing = safe_load('path_to_bing_data', "Bing")
nrc = safe_load('path_to_nrc_data', "NRC")
afinn = safe_load('path_to_afinn_data', "AFINN")

# Load NLTK opinion lexicon
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())
nltk_opinion = pd.DataFrame({
    'word': list(positive_words) + list(negative_words),
    'sentiment': ['positive'] * len(positive_words) + ['negative'] * len(negative_words),
    'dict': 'nltk_opinion_lexicon'
})

# Combine dictionaries
dicts = pd.concat([df for df in [huliu, bing, nrc, afinn, nltk_opinion] if not df.empty])
if dicts.empty:
    print("No lexicons were successfully loaded.")
else:
    dicts['word'] = lemmatize_words(dicts['word'].tolist())
    dicts = dicts.drop_duplicates()

# Function to calculate sentiment scores
def calculate_sentiment_scores(text):
    words = text.lower().split()
    lemmatized_words = lemmatize_words(words)
    scores = {}

    for dict_name in dicts['dict'].unique():
        dict_words = dicts[dicts['dict'] == dict_name]
        positive_words = set(dict_words[dict_words['sentiment'] == 'positive']['word'])
        negative_words = set(dict_words[dict_words['sentiment'] == 'negative']['word'])

        positive_count = sum(word in positive_words for word in lemmatized_words)
        negative_count = sum(word in negative_words for word in lemmatized_words)
        total_count = len(lemmatized_words)

        scores[f'{dict_name}_positive'] = positive_count / total_count if total_count > 0 else 0
        scores[f'{dict_name}_negative'] = negative_count / total_count if total_count > 0 else 0
        scores[f'{dict_name}_net'] = (positive_count - negative_count) / total_count if total_count > 0 else 0

    # Add VADER sentiment
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(text)
    scores.update({f'vader_{k}': v for k, v in vader_scores.items()})

    return scores

# Calculate z-scores and correlation matrix
def compute_z_scores_and_correlations(df, columns):
    for column in columns:
        df[f'{column}_z'] = (df[column] - df[column].mean()) / df[column].std()
        
    z_columns = [f'{col}_z' for col in columns]
    correlation_matrix = df[z_columns].corr()
    return df, correlation_matrix


def get_sentiment(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that only responds in valid JSON format. Your task is to provide sentiment analysis and present the results, model into a specific JSON structure."},
                {"role": "user", "content": f"Help me to get the sentiment analysis of the text from 1-10 into JSON files without any ''' json ''' that is not in the string format, where the text is stored in a variable called text: {text}"}
            ]
        )

        # Extract the content from the response
        content = response.choices[0].message.content

        # Parse the JSON content
        sentiment_data = json.loads(content)
        return sentiment_data

    except Exception as e:
        print(f"Error processing text: {text[:50]}... Error: {str(e)}")
        return None

# Function to apply sentiment analysis with rate limiting and iteration
def apply_sentiment_analysis(df):
    results = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        sentiment_list = []
        for _ in range(5):  # Iterate 5 times for each text
            sentiment = get_sentiment(row['text'])
            sentiment_list.append(sentiment)
            time.sleep(1)  # Rate limiting: wait 1 second between requests
        results.append(sentiment_list)
    return results

# def convert_list_to_float(score_list):
#     return [float(score) for score in score_list]

# Function to convert the list to floats, ignoring None and non-integer/float values
def convert_list_to_float(score_list):
    return [float(score) for score in score_list if isinstance(score, (int, float)) and not np.isnan(score)]


def extract_sentiment_scores(sentiment_list):
    scores = []
    for item in sentiment_list:
        try:
            if 'sentiment_analysis' in item and 'sentiment_score' in item['sentiment_analysis']:
                scores.append(item['sentiment_analysis']['sentiment_score'])
            elif 'sentiment_analysis' in item and 'overall_sentiment' in item['sentiment_analysis']:
                scores.append(item['sentiment_analysis']['overall_sentiment'])
            elif 'sentiment_analysis' in item and 'score' in item['sentiment_analysis']:
                scores.append(item['sentiment_analysis']['score'])
            elif 'sentiment' in item and 'subjectivity_confidence' in item['sentiment']:
                scores.append(item['sentiment']['subjectivity_confidence'] * 10)
            elif 'sentiment' in item and 'score' in item['sentiment']:
                scores.append(item['sentiment']['score'])
            elif 'sentiment' in item and 'positive' in item['sentiment']:
                scores.append(item['sentiment']['positive'])
            elif 'sentiment' in item and 'overall' in item['sentiment']:
                scores.append(item['sentiment']['overall'])
            elif 'sentiment' in item and 'sentiment_score' in item['sentiment']:
                scores.append(item['sentiment']['sentiment_score'])
            elif 'sentiment_score' in item:
                scores.append(item['sentiment_score'])
            elif 'sentiment' in item and 'overall_score' in item['sentiment']:
                scores.append(float(item['sentiment']['overall_score']))
            else:
                scores.append(None)  # Append None if no score is found
        except (KeyError, TypeError, ValueError):
            scores.append(None)  # Append None in case of any exception
    
    
    return scores

# Assuming df is your DataFrame with a 'text' column
# Apply the sentiment analysis


# # Process each URL in the list
# def openai_sentiment_analysis(df,csv_output,image_output):
#     df['sentiment_iterate'] = apply_sentiment_analysis(df)

#     # Apply the function to the DataFrame
#     df['sentiment_scores'] = df['sentiment_iterate'].apply(extract_sentiment_scores)
#     # Apply the conversion to the entire column
#     df['sentiment_scores_number'] = df['sentiment_scores'].apply(convert_list_to_float)

#     # Calculate mean and std for each row, ignoring empty lists
#     df['openai_mean_sentiment'] = df['sentiment_scores_number'].apply(lambda x: np.mean(x) if x else None) / 10
#     df['std_sentiment'] = df['sentiment_scores_number'].apply(lambda x: np.std(x) if x else None) / 10

#     # Calculate overall mean and std (ignoring None in the DataFrame)
#     overall_mean = df['openai_mean_sentiment'].mean()
#     overall_std = df['std_sentiment'].mean()

#     print(f"Overall Mean of sentiment scores: {overall_mean:.2f}")
#     print(f"Overall Standard Deviation of sentiment scores: {overall_std:.2f}")

#     # If you want the std of all scores combined (not averaged per row)
#     all_scores = [score for scores in df['sentiment_scores_number'] for score in scores]
#     combined_std = np.std(all_scores) / 10
#     print(f"Combined Standard Deviation of all scores: {combined_std:.2f}")

#     # Function to compute z-scores and correlations
#     def compute_z_scores_and_correlations(df, columns):
#         for column in columns:
#             df[f'{column}_z'] = (df[column] - df[column].mean()) / df[column].std()
            
#         z_columns = [f'{col}_z' for col in columns]
#         correlation_matrix = df[z_columns].corr()
#         return df, correlation_matrix

#     # Define metrics to be analyzed
#     metrics = ['nltk_opinion_lexicon_net', 'vader_neg', 
#             'vader_neu', 'vader_pos', 'vader_compound', 'openai_mean_sentiment']

#     # Compute z-scores and correlation matrix
#     df, correlation_matrix = compute_z_scores_and_correlations(df, metrics)

#     # Plot the correlation matrix
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
#     plt.title('Correlation Matrix of Sentiment Measures')
#     plt.savefig(image_output, dpi=300)
#     plt.show()

#     # Save high-resolution correlation matrix plot
#     return df

import json

def openai_sentiment_analysis(df, csv_output, image_output, json_output_path):
    # Apply sentiment analysis functions
    df['sentiment_iterate'] = apply_sentiment_analysis(df)
    df['sentiment_scores'] = df['sentiment_iterate'].apply(extract_sentiment_scores)
    df['sentiment_scores_number'] = df['sentiment_scores'].apply(convert_list_to_float)
    
    # Calculate mean and std for each row, ignoring empty lists
    df['openai_mean_sentiment'] = df['sentiment_scores_number'].apply(lambda x: np.mean(x) if x else None) / 10
    df['std_sentiment'] = df['sentiment_scores_number'].apply(lambda x: np.std(x) if x else None) / 10

    # Calculate overall mean and std (ignoring None in the DataFrame)
    overall_mean = df['openai_mean_sentiment'].mean()
    overall_std = df['std_sentiment'].mean()

    print(f"Overall Mean of sentiment scores: {overall_mean:.2f}")
    print(f"Overall Standard Deviation of sentiment scores: {overall_std:.2f}")

    # Calculate combined standard deviation of all scores
    all_scores = [score for scores in df['sentiment_scores_number'] for score in scores]
    combined_std = np.std(all_scores) / 10
    print(f"Combined Standard Deviation of all scores: {combined_std:.2f}")
    
    # Function to compute z-scores and correlations
    def compute_z_scores_and_correlations(df, columns):
        for column in columns:
            df[f'{column}_z'] = (df[column] - df[column].mean()) / df[column].std()
            
        z_columns = [f'{col}_z' for col in columns]
        correlation_matrix = df[z_columns].corr()
        return df, correlation_matrix

    # Define metrics to be analyzed
    metrics = ['nltk_opinion_lexicon_net', 'vader_neg', 
               'vader_neu', 'vader_pos', 'vader_compound', 'openai_mean_sentiment']

    # Compute z-scores and correlation matrix
    df, correlation_matrix = compute_z_scores_and_correlations(df, metrics)

    # Plot and save the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Sentiment Measures')
    plt.savefig(image_output, dpi=300)
    plt.show()
    
    # Prepare JSON data structure
    sentiment_data = {
        "overall_mean_sentiment": overall_mean,
        "overall_standard_deviation": overall_std,
        "combined_standard_deviation": combined_std,
        "segments": []
    }

    for index, row in df.iterrows():
        segment = {
            "id": index,
            "nltk_opinion_lexicon_net": row['nltk_opinion_lexicon_net'],
            "vader_neg": row['vader_neg'],
            "vader_neu": row['vader_neu'],
            "vader_pos": row['vader_pos'],
            "vader_compound": row['vader_compound'],
            "openai_mean_sentiment": row['openai_mean_sentiment'],
            "std_sentiment": row['std_sentiment'],
            "tokens": row['sentiment_scores']
        }
        sentiment_data["segments"].append(segment)
    
    # Export JSON data to file
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(sentiment_data, json_file, ensure_ascii=False, indent=4)
    
    print(f"Sentiment data successfully written to {json_output_path}")
    df.to_csv(csv_output)
    # Return the DataFrame for further inspection or use
    return "Process Successful"

# # Example Usage
# csv_file_path = 'sample_data.csv'
# csv_output = 'sample_output.csv'
# image_output = 'correlation_matrix.png'
# json_output_path = 'sentiment_results.json'

# # Assuming df is predefined
# # df = pd.read_csv(csv_file_path)
# openai_sentiment_analysis(df, csv_output, image_output, json_output_path)

# import ast

# def csv_to_json(df, json_output_path):
#     # Read the CSV file into DataFrame
#     # df = pd.read_csv(csv_file_path)
  
#     # Concatenate all text values to create the transcription_text
#     # transcription_text = " ".join(df['text'].tolist())
#     assert 'text' in df.columns, "Column 'text' is missing"
#     assert 'tokens' in df.columns, "Column 'tokens' is missing"

#     # Convert 'tokens' from string representation to lists
#     df['tokens'] = df['tokens'].apply(ast.literal_eval)

#     # Concatenate all text values to create the transcription_text
#     transcription_text = " ".join(df['text'].tolist())

#     # Initialize the transcription_data dictionary
#     transcription_data = {
#         "transcription_text": transcription_text,
#         "segments": []
#     }

#     # Iterate through each row to create a segment dictionary
#     for index, row in df.iterrows():
#         segment = {
#             "id": row['segment_id'],
#             "seek": row['seek'],
#             "start": row['start'],
#             "end": row['end'],
#             "text": row['text'],
#             "tokens": row['tokens'],
#             "temperature": row['temperature'],
#             "avg_logprob": row['segment_avg_logprob'],
#             "compression_ratio": row['segment_compression_ratio'],
#             "no_speech_prob": row['segment_no_speech_prob']
#         }
#         transcription_data["segments"].append(segment)
    
#     # Convert the transcription_data dictionary to a JSON string
#     transcription_json = json.dumps(transcription_data, ensure_ascii=False, indent=4)
  
#     # Write the JSON string to file
#     with open(json_output_path, 'w', encoding='utf-8') as json_file:
#         json_file.write(transcription_json)
    
#     print("Process successful")

# # Usage
# csv_file_path = 'transcription_result_temp.csv'
# json_output_path = 'transcription_result.json'
# csv_to_json(csv_file_path, json_output_path)

for url in urls:
    folder_name, audio_file_path = process_video(url)
    # folder_name = os.path.join('public', 'packages', folder_name)

    if folder_name and audio_file_path:
        output_json_path = os.path.join(folder_name, 'transcription_result.json')
        # Ensure the file exists before attempting transcription
        if os.path.exists(audio_file_path):
            transcription_results = transcribe_audio(audio_file_path)
            save_transcription_to_json(transcription_results, output_json_path)
            df = convert_json_to_dataframe(folder_name, output_json_path)

            tqdm.pandas()

            sentiment_scores = df["text"].progress_apply(calculate_sentiment_scores)
            sentiment_df = pd.DataFrame(sentiment_scores.tolist())
            sample = pd.concat([df, sentiment_df], axis=1)
            output_csv_path = os.path.join(folder_name, 'transcription_result.csv')
            sample.to_csv(output_csv_path)

            # Create the time series DataFrame
            time_series_df = create_time_series(sample)
            # Plot the metrics
            output_png_path = os.path.join(folder_name, 'metrics_plot.png')
            plot_metrics(time_series_df, output_png_path)

            output_csv_path = os.path.join(folder_name, 'transcription_result_temp.csv')
            output_png_path = os.path.join(folder_name, 'correlation_matrix.png')

            output_json_path_new = os.path.join(folder_name, 'transcription_result_new.json')
            openai_sentiment_analysis(sample,output_csv_path,output_png_path,output_json_path_new)
        else:
            print(f"The audio file '{audio_file_path}' does not exist.")

    print("\n--- Processing next video ---\n")