import os
import random
from collections import defaultdict
import nltk
from nltk.util import ngrams

# Step 1: Import necessary libraries
nltk.download('punkt')

# Step 2: Read and preprocess the text from the files
def read_text_files(folder_path):
    text_corpus = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                text = text.lower()  # Convert to lowercase
                text_corpus.extend(nltk.sent_tokenize(text))
    return text_corpus

# Step 3: Create bigram and trigram models
def create_ngram_models(text_corpus, n):
    ngram_model = defaultdict(list)

    for sentence in text_corpus:
        words = nltk.word_tokenize(sentence)
        ngrams_list = list(ngrams(words, n, pad_left=True, pad_right=True))

        for ngram in ngrams_list:
            ngram_model[ngram[:-1]].append(ngram[-1])

    return ngram_model

folder_path = 'C:/Users/Lenovo/OneDrive/Desktop/IR/dhruv'  # Update with your folder path
text_corpus = read_text_files(folder_path)
bigram_model = create_ngram_models(text_corpus, 2)
trigram_model = create_ngram_models(text_corpus, 3)

# Step 4: Generate random text using the models
def generate_random_text(model, n, num_sentences=5, max_length=100):
    generated_text = []

    for _ in range(num_sentences):
        sentence = []
        current_ngram = [None] * (n - 1)
        
        while True:
            next_word = random.choice(model[tuple(current_ngram)])
            if next_word is None:
                break
            sentence.append(next_word)
            current_ngram = current_ngram[1:] + [next_word]
            if len(sentence) >= max_length:
                break
        
        generated_text.append(' '.join(sentence))

    return '\n'.join(generated_text)

# Generate random bigram text
random_bigram_text = generate_random_text(bigram_model, 2)

# Generate random trigram text
random_trigram_text = generate_random_text(trigram_model, 3)

print("Random Bigram Text:")
print(random_bigram_text)

print("\nRandom Trigram Text:")
print(random_trigram_text)