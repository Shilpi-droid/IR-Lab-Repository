from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data

# Text Preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Tokenization, lowercase conversion, and stopword removal
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [ps.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

preprocessed_documents = [preprocess_text(doc) for doc in documents]

# Create the term-document matrix using TfidfVectorizer
vectorizer = TfidfVectorizer()
term_doc_matrix = vectorizer.fit_transform(preprocessed_documents)

# Apply SVD on the term-document matrix
num_topics = 100
svd = TruncatedSVD(n_components=num_topics)
lsi_matrix = svd.fit_transform(term_doc_matrix)

# Print the top terms for each topic
terms = vectorizer.get_feature_names_out()
for i, topic in enumerate(svd.components_):
    top_terms_idx = topic.argsort()[-10:][::-1]
    top_terms = [terms[idx] for idx in top_terms_idx]
    print(f"Topic {i + 1}: {', '.join(top_terms)}")

# Example query
query = "science and technology"

# Preprocess the query
preprocessed_query = preprocess_text(query)

# Transform the query using the same LSI model
query_vector = vectorizer.transform([preprocessed_query])
query_lsi = svd.transform(query_vector)

print(query_lsi)

# Compute cosine similarities between the query and documents
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarities = cosine_similarity(query_lsi, lsi_matrix)
print(cosine_similarities)
# Get the index of the most relevant document
most_relevant_doc_index = cosine_similarities.argmax()

# Print the most relevant document
print("Most relevant document:")
print(documents[most_relevant_doc_index])

