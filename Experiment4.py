# Import necessary libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the 20 newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X = newsgroups.data
y = newsgroups.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Convert text data to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the K-Nearest Neighbors classifier
k = 5  # You can change the value of k as needed
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
knn_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test_tfidf)

# Calculate accuracy and print the results
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

from sklearn.neighbors import NearestCentroid
# Initialize the Rocchio classifier
rocchio_classifier = NearestCentroid()

# Train the classifier
rocchio_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = rocchio_classifier.predict(X_test_tfidf)

# Calculate accuracy and print the results
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

from sklearn.naive_bayes import MultinomialNB
# Initialize the Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()

# Train the classifier
naive_bayes_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test_tfidf)

# Calculate accuracy and print the results
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

from sklearn.metrics import f1_score

# Initialize and train the Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tfidf, y_train)
y_pred_naive_bayes = naive_bayes_classifier.predict(X_test_tfidf)

# Calculate F1-score for Naive Bayes
f1_score_naive_bayes = f1_score(y_test, y_pred_naive_bayes, average='weighted')

# Initialize and train the Nearest Centroid classifier
rocchio_classifier = NearestCentroid()
rocchio_classifier.fit(X_train_tfidf, y_train)
y_pred_rocchio = rocchio_classifier.predict(X_test_tfidf)

# Calculate F1-score for Nearest Centroid (Rocchio)
f1_score_rocchio = f1_score(y_test, y_pred_rocchio, average='weighted')

# Initialize and train the K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
knn_classifier.fit(X_train_tfidf, y_train)
y_pred_knn = knn_classifier.predict(X_test_tfidf)

# Calculate F1-score for K-Nearest Neighbors
f1_score_knn = f1_score(y_test, y_pred_knn, average='weighted')

print(f'F1-score (Naive Bayes): {f1_score_naive_bayes:.2f}')
print(f'F1-score (Rocchio): {f1_score_rocchio:.2f}')
print(f'F1-score (K-Nearest Neighbors): {f1_score_knn:.2f}')
