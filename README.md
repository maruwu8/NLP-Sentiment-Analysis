# NLP-Sentiment-Analysis
This project features various machine learning NLP algorithms (LSTM, SVMs, Naive Bayes) and a mini app that extracts and performs sentiment analysis on blog posts using the Tumblr API.

This project involves the utilization of various machine learning algorithms for natural language processing (NLP) tasks, including LSTM, SVMs, and a baseline Naive Bayes classifier. The primary objective is to develop a sentiment analysis application that leverages the Tumblr API. The application, created using Streamlit, incorporates essential libraries such as scikit-learn, TensorFlow, and Keras for model training and deployment.

For data analysis, the project employs the Sentiment140 dataset sourced from Kaggle, which consists of 1.6 million tweets. This dataset serves as the foundation for training and evaluating the sentiment analysis models. Additionally, the project utilizes Stanford's GloVe 100-dimensional word embeddings file, glove.6B.100d.txt, which is obtained from Kaggle. These pre-trained word embeddings enhance the models' understanding of semantic relationships between words.

By combining these machine learning algorithms, libraries, and datasets, the sentiment analysis application extracts Tumblr blog posts, performs sentiment analysis, and categorizes them as either positive or negative based on the trained models.

Naive Bayes Classifier (NLP): In natural language processing (NLP), the Naive Bayes classifier is a probabilistic algorithm used for text classification tasks. It assumes that the presence or absence of each word in a document is independent of other words, given the class label. Naive Bayes classifiers are commonly used in NLP for tasks like sentiment analysis, spam detection, and document categorization.

LSTM (Long Short-Term Memory) (NLP): LSTM is a type of recurrent neural network (RNN) architecture that is widely used in NLP tasks. It is designed to handle sequential data and has the ability to capture long-term dependencies in text. LSTMs are effective in tasks like language modeling, machine translation, sentiment analysis, and named entity recognition. They can process input text sequentially, retaining information from previous words to understand the context and make predictions.

SVM (Support Vector Machine) (NLP): SVM is a machine learning algorithm that can be used for text classification and sentiment analysis in NLP. It finds an optimal hyperplane to separate different classes of text data. SVMs in NLP are often used with features derived from text, such as bag-of-words or TF-IDF vectors. They can handle high-dimensional feature spaces and are effective in tasks like sentiment analysis, text categorization, and topic classification.

In NLP, Naive Bayes classifiers are commonly used for their simplicity and speed, LSTM models are suitable for capturing sequential dependencies in text, and SVMs are versatile algorithms capable of handling high-dimensional text features for classification tasks.

After training the models, we proceed to evaluate their performance using various evaluation measures. The following parameters are utilized to assess the models' performance:

Accuracy Score: The accuracy score provides an indication of the overall correctness of the predictive models. A high accuracy score, typically above 90%, suggests that the models perform well in classifying the blog posts.

ROC-AUC Curve: The ROC-AUC curve is a graphical representation of the model's ability to distinguish between the positive and negative classes. The Area Under the Curve (AUC) is calculated to summarize the ROC curve. A higher AUC value signifies a better performance in class separation.

Confusion Matrix with Plot: The confusion matrix is a square matrix of size N x N, where N represents the number of target classes. It allows us to evaluate the performance of the classification models by comparing the actual target values with the predicted values. Additionally, a visual plot of the confusion matrix provides a clearer understanding of the model's performance.

By analyzing these evaluation measures, we can assess the effectiveness of the models in accurately classifying the sentiment of the extracted Tumblr blog posts.
