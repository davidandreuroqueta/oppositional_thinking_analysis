import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from functools import partial
from transformers import DebertaTokenizer
from sklearn.decomposition import TruncatedSVD
import gensim.downloader as api # works with scipy==1.10.1
import string
from transformers import RobertaModel, RobertaTokenizer
from gensim.models.keyedvectors import KeyedVectors
import os, sys
import logging
import torch


# Load dataloaders
sys.path.append("pan-clef-2024-oppositional/")
from data_tools.dataset_loaders import load_dataset_classification

nltk.download('punkt')
nltk.download('stopwords')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to preprocess text
def preprocess_text(text: str,
                    stop_words=stopwords.words('english')):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Remove stopwords, punctuation marks, and symbols
    tokens = [w for w in tokens
              if w not in stop_words
              and w not in string.punctuation
              and w.isalnum()]
    
    return tokens

# Function to calculate average vector representation of a document
def document_vector(tokens: list[str],
                    word_vectors: KeyedVectors):
    # Initialize empty vector
    document_vector = np.zeros((300,))
    # Count number of words in document
    words_count = 0
    # Iterate over each word in the document
    for word in tokens:
        if word in word_vectors.key_to_index:
            # Add word vector to document vector
            document_vector += word_vectors[word]
            words_count += 1
    # Average the document vector
    if words_count != 0:
        document_vector /= words_count
    return document_vector

# Function to create LSA features
def lsa_embedding(train_texts: pd.DataFrame,
                  val_texts: pd.DataFrame,
                  n_components: int=100,
                  process_text: bool=True,
                  stop_words=stopwords.words('english')):
    
    if process_text:
        # Define the custom preprocess function with stop words
        custom_preprocess = partial(preprocess_text, stop_words=stop_words
                                        if stop_words 
                                        else stopwords.words('english'))

        # Preprocess text
        train_texts['tokens_lsa'] = train_texts['text'].apply(custom_preprocess) \
            .apply(' '.join)
        val_texts['tokens_lsa'] = val_texts['text'].apply(custom_preprocess) \
            .apply(' '.join)
    else:
        train_texts['tokens_lsa'] = train_texts['text']
        val_texts['tokens_lsa'] = val_texts['text']
        

    
    # Create TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer(binary=False,
                                       ngram_range=(3, 3),
                                       analyzer='char')
    train_tfidf = tfidf_vectorizer.fit_transform(train_texts['tokens_lsa'])
    val_tfidf = tfidf_vectorizer.transform(val_texts['tokens_lsa'])
    
    # Apply SVD
    svd = TruncatedSVD(n_components=n_components)
    train_lsa = svd.fit_transform(train_tfidf)
    val_lsa = svd.transform(val_tfidf)
    
    # Add LSA embeddings to the dataframes
    train_texts['lsa_embedding'] = list(train_lsa)
    val_texts['lsa_embedding'] = list(val_lsa)
    
    return train_texts, val_texts


def fasttext_embedding(word_vectors: KeyedVectors,
                       train_texts: pd.DataFrame,
                       val_texts: pd.DataFrame,
                       process_text: bool=True,
                       stop_words=stopwords.words('english')):
    
    if process_text:
        # Define the custom preprocess function with stop words
        custom_preprocess = partial(preprocess_text, stop_words=stop_words
                                        if stop_words 
                                        else stopwords.words('english'))

        # Preprocess text
        train_texts['tokens_fasttext'] = train_texts['text'].apply(custom_preprocess)
        val_texts['tokens_fasttext'] = val_texts['text'].apply(custom_preprocess)
    else:
        train_texts['tokens_fasttext'] = train_texts['text']
        val_texts['tokens_fasttext'] = val_texts['text']
        
            
    word_vectors = api.load('fasttext-wiki-news-subwords-300')
    
    # TRAIN TEXTS
    # Create dictionary to store document vectors
    document_vectors = {}

    # Iterate over each document in the dataset
    for index, row in train_texts.iterrows():
        # Calculate document vector
        doc_vector = document_vector(row['tokens_fasttext'], word_vectors)
        # Store document vector in dictionary
        document_vectors[index] = doc_vector
    
    # Add document vectors as a new column in the DataFrame
    train_texts['fasttext_embeeding'] = document_vectors
    
    # VAL TEXTS
    # Create dictionary to store document vectors
    document_vectors = {}

    # Iterate over each document in the dataset
    for index, row in val_texts.iterrows():
        # Calculate document vector
        doc_vector = document_vector(row['tokens_fasttext'], word_vectors)
        # Store document vector in dictionary
        document_vectors[index] = doc_vector
    
    # Add document vectors as a new column in the DataFrame
    val_texts['fasttext_embeeding'] = document_vectors 
    
    return train_texts, val_texts


import torch
from torch.utils.data import DataLoader, TensorDataset

def roberta_embedding(train_texts: pd.DataFrame,
                      val_texts: pd.DataFrame,
                      lang: str='en',
                      batch_size=16):
    # Carga el modelo y el tokenizer de RoBERTa
    if lang == 'es':
        model = RobertaModel.from_pretrained('bertin-project/bertin-roberta-base-spanish')
        tokenizer = RobertaTokenizer.from_pretrained('bertin-project/bertin-roberta-base-spanish')
    elif lang == 'en':
        model = RobertaModel.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        raise ValueError("Invalid language. Expected 'en' or 'es'.")

    # Mueve el modelo a la GPU si está disponible
    model.to(device)
    model.eval()  # Set model to evaluation mode to disable dropout layers

    # Función para obtener embeddings en batches utilizando DataLoader
    def get_embeddings(texts):
        # Tokeniza los textos y crea un dataset de PyTorch
        inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        data_loader = DataLoader(dataset, batch_size=batch_size)

        # Lista para almacenar los embeddings
        embeddings = []

        # Desactiva el cálculo de gradientes para ahorrar memoria y aumentar la velocidad
        with torch.no_grad():
            for batch in data_loader:
                logging.info(f"Processing batch {len(embeddings) + 1} of {len(data_loader)}...")
                input_ids, attention_mask = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs[0][:, 0, :].detach().cpu().numpy()  # Solo se toma el embedding del token CLS
                embeddings.append(batch_embeddings)

        # Concatena los embeddings de todos los batches
        return np.concatenate(embeddings, axis=0)

    # Calcula los embeddings para train_texts y val_texts
    logging.info("Processing training data embeddings...")
    train_embeddings = get_embeddings(train_texts['text'].tolist())
    logging.info("Processing validation data embeddings...")
    val_embeddings = get_embeddings(val_texts['text'].tolist())

    # Añade los embeddings de RoBERTa a los DataFrames
    train_texts['roberta_embedding'] = list(train_embeddings)
    val_texts['roberta_embedding'] = list(val_embeddings)

    return train_texts, val_texts



def all_embedings(train: pd.DataFrame,
                  val: pd.DataFrame,
                  lang: str,
                  word_vectors: KeyedVectors):
    if lang == 'en':
        stop_words = stopwords.words('english')
    elif lang == 'es':
        stop_words = stopwords.words('spanish')
    else:
        raise ValueError("Invalid language. Expected 'en' or 'es'.")
    
    try:
        train, val = lsa_embedding(train, val, stop_words=stop_words)
    except Exception as e:
        logging.error(f"Error calculating LSA embeddings: {e}")
    
    try:
        train, val = fasttext_embedding(word_vectors, train, val, stop_words=stop_words)
    except Exception as e:
        logging.error(f"Error calculating FastText embeddings: {e}")
    
    # try:
    #     train, val = roberta_embedding(train, val, lang=lang)
    # except Exception as e:
    #     logging.error(f"Error calculating RoBERTa embeddings: {e}")
    
    return train, val

# Init logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read files
TRAIN_DATASET_ES = "Dataset-Oppositional/training/dataset_es_train.json"
TRAIN_DATASET_EN = "Dataset-Oppositional/training/dataset_en_train.json"
TEST_DATASET_EN = "Dataset-Oppositional/test/dataset_en_official_test_nolabels.json"
TEST_DATASET_ES = "Dataset-Oppositional/test/dataset_es_official_test_nolabels.json"

logging.info("Loading datasets...")
try:
    texts_es, labels_es, ids_es = load_dataset_classification('es', string_labels=False, positive_class='conspiracy')
    texts_en, labels_en, ids_en = load_dataset_classification('en', string_labels=False, positive_class='conspiracy')
    df_es = pd.DataFrame({'id': ids_es, 'text': texts_es, 'label': labels_es})
    df_en = pd.DataFrame({'id': ids_en, 'text': texts_en, 'label': labels_en})
    # Change row id to 'id' column
    df_es.index = df_es['id']
    df_en.index = df_en['id']
    
except Exception as e:
    logging.error(f"Error loading datasets: {e}")
    sys.exit(1)
    
logging.info(f'Processing with: {device}')

# word_vectors = api.load('fasttext-wiki-news-subwords-300')
word_vectors = None
# Split df_en in train and val
train_en, val_en = train_test_split(df_en, test_size=0.2, random_state=50)
train_es, val_es = train_test_split(df_es, test_size=0.2, random_state=50)

logging.info("Calculating english embeddings...")
train_en, val_en = all_embedings(train_en, val_en, 'en', word_vectors=word_vectors)
logging.info("Done")

logging.info("Calculating spanish embeddings...")
train_es, val_es = all_embedings(train_es, val_es, 'es', word_vectors=word_vectors)
logging.info("Done")

# Save train_en, val_en, train_es, val_es
logging.info("Saving datasets...")
train_en.to_csv('processed_embeddings/train_en.csv', index=False, sep=';', encoding='utf-8')
val_en.to_csv('processed_embeddings/val_en.csv', index=False, sep=';', encoding='utf-8')
train_es.to_csv('processed_embeddings/train_es.csv', index=False, sep=';', encoding='utf-8')
val_es.to_csv('processed_embeddings/val_es.csv', index=False, sep=';', encoding='utf-8')