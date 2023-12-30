1 Background:
In the era of digital information, news articles are generated at an unprecedented rate. Analyzing
and categorizing these articles are crucial tasks for various applications, such as personalized content
recommendations, sentiment analysis, and trend prediction. Traditional methods of news classification
often relied on manually constructed feature sets and shallow machine learning algorithms. However,
with the coming of deep learning techniques, especially Transformer-based models like BERT
(Bidirectional Encoder Representations from Transformers) and RoBERTa (Robustly optimized BERT
approach), the landscape of natural language processing (NLP) tasks, including news classification,
has undergone a significant transformation.
Before delving into Transformer-based algorithms for news classification, it’s essential to understand
the context of this technology.
1.1 Traditional NLP Challenges:
• Feature Engineering: Traditional NLP algorithms heavily relied on manual feature engineer-
ing. Linguistic experts crafted features like bag-of-words, TF-IDF (Term Frequency-Inverse
Document Frequency), and word embeddings to represent text data.
• Limited Context Understanding: Earlier models struggled with capturing contextual informa-
tion of a language. They couldn’t understand the meaning of words in relation to surrounding
words, leading to limitations in tasks like sentiment analysis and news classification.
1.2 Introduction of Transformer Architecture:
• Attention Mechanism: Transformers introduced the concept of self-attention mechanism,
enabling the model to weigh the importance of different words in a sentence concerning
one another. This mechanism allowed the model to capture long-range dependencies and
contextual information effectively.
• BERT and Pretrained Models: BERT, introduced by Google in 2018, utilized transformers
for pretraining on massive corpora. The model learned contextual representations of words,
making it highly effective for various downstream NLP tasks without task-specific feature
engineering.
1.3 BERT and RoBERTa in NLP Tasks:
• BERT: Introduced by Google in 2018, BERT brought about a major shift in NLP. Unlike
previous models, BERT reads text bi-directionally, considering both the left and right context
in all layers of the neural network. This bidirectional approach allows BERT to capture the
rich contextual information of words, and understand polysemy (multiple meanings of a
word) and homonymy (same spelling, different meanings). BERT is pretrained on a massive
corpus, learning to predict missing words in a sentence. This pretraining provides BERT
with a deep understanding of language, making it capable of tasks such as text classification,
named entity recognition, and question answering.
• RoBERTa: RoBERTa, an improvement upon BERT, was developed by Facebook AI in 2019.
It optimizes the pretraining process, addressing BERT’s limitations. RoBERTa removes
1
the Next Sentence Prediction (NSP) task from pretraining, allowing the model to focus
solely on the masked language model (MLM) objective. Additionally, RoBERTa uses
larger mini-batches and trains on more data, enhancing its generalization and robustness.
These modifications lead to a more nuanced understanding of context, making RoBERTa
particularly effective in tasks requiring deep comprehension of textual data.
1.4 News Classification with BERT and RoBERTa:
News classification with BERT and RoBERTa involves leveraging the power of these pre-trained
transformer-based models to automatically categorize news articles into specific topics or classes.
This process significantly enhances the efficiency and accuracy of news organization, aiding in tasks
like content recommendation, sentiment analysis, and trend analysis. Here’s a detailed explanation of
how these models are applied in news classification:
1. Pretraining: BERT and RoBERTa are pretrained on large corpora, learning the contextual relation-
ships between words in sentences. During this phase, the models grasp intricate details of language,
enabling them to understand the underlying meaning of words and phrases within various contexts.
2. Fine-Tuning: After pretraining, these models are fine-tuned on a specific news dataset. Fine-tuning
involves training the models on a smaller, domain-specific dataset related to news articles. During this
process, the models adjust their parameters to better fit the text seen in the news domain. Fine-tuning
is crucial as it allows the models to specialize in classifying news articles accurately.
3. Text Representation: When a news article is fed into the fine-tuned BERT or RoBERTa model, the
text undergoes tokenization, breaking it down into smaller units called tokens. These tokens are then
converted into high-dimensional vectors, capturing the semantic meaning and context of the words.
The bidirectional nature of these models ensures that the context of each word is considered, leading
to a more profound understanding of the article.
4. Classification: Once the news article is represented as vectors, these vectors are fed into a
classification layer. The classification layer maps the high-dimensional representations to specific
categories or labels, determining the topic of the news article. For instance, categories could include
politics, sports, entertainment, technology, etc.
5. Benefits:
• Contextual Understanding: BERT and RoBERTa excel at understanding the subtle contextual
details in news articles, making them proficient at distinguishing between closely related
topics.
• Handling Ambiguity: News articles often contain ambiguous language. BERT and RoBERTa
can navigate through this ambiguity, providing accurate classifications even in complex
scenarios.
• Continuous Learning: As transformer-based models, BERT and RoBERTa can be continually
fine-tuned on new data, adapting to changing language patterns and ensuring up-to-date and
relevant classifications.
5. Intuition: As news classification is crucial for many applications like recommendation and senti-
ment analysis, we sought advanced NLP techniques to enhance accuracy. Transformer-based models
like BERT and RoBERTa have revolutionized language understanding through their bidirectional, con-
textual pretraining. Their ability to grasp semantics and handle ambiguity makes them well-suited for
categorizing complex news data. Fine-tuning them on our dataset would allow these powerful models
to specialize in classifying news specifically. We hypothesize these techniques will significantly
outperform previous shallow methods, providing state-of-the-art news classification performance to
enable more intelligent information analysis applications.
In summary, employing BERT and RoBERTa for news classification involves harnessing their deep
contextual understanding through fine-tuning techniques, allowing accurate categorization of news
articles across diverse topics. This application highlights their versatility in processing human
language, serving as invaluable tools for information analysis. Transformer-based models, such
as BERT and RoBERTa, significantly enhance news classification accuracy by deciphering subtle
language details, enabling effective categorization even with complex or ambiguous topics. Their
2
introduction marks a transformative shift in NLP and news classification, elevating the accuracy of
tasks and opening doors for sophisticated applications in natural language understanding.
2 Methods
1. Exploratory Data Analysis (EDA):
In the context of news article classification, a thorough EDA helped us gain insights into the nature of
the data, making it easier to preprocess effectively and design a robust model. We also attempt to
analyze characteristics of features like “author”, “headline” (news article caption), and “description”
(news article description) to give us better insight into whether they can be used for training.
The various methods used are:
• Text Length Analysis: We analyzed the distribution of text lengths of the attributes “headline”
and “description” for the different classes in the dataset. This helped us decide on an
appropriate maximum sequence length for tokenization and compared the text lengths across
different categories to see if certain categories tend to have longer or shorter articles.
• Class Distribution: We examined the distribution of news articles across different
classes/categories to ensure the dataset is balanced or if we needed to use some techniques
like over or undersampling to balance it.
• Word Frequency Analysis:Conducting word frequency analysis on the entire dataset and
within each category aided in understanding the vocabulary and identifying potential stop
words or special characters requiring handling during text preprocessing. This analysis
also provided insights into the types of headlines the model is likely to classify accurately.
For instance, in the "politics" class, the analysis revealed high frequencies of the keywords
"clinton" and "trump" (after conversion to lowercase). This suggests that models trained on
this dataset are inclined to classify news headlines containing these keywords as belonging
to the "politics" category.
• Visualizations: Created visualizations such as word clouds, bar charts, or heatmaps to
represent word frequencies, class distributions, text lengths, or any other relevant information.
This helped in revealing patterns that were not immediately obvious from raw data.
• Analysis of the relationship between dependent and independent attributes in the dataset:
We tried to identify whether there was any useful relationship between the ’author’ field and
the ’category’ field (class variable). If there was one, we could have used the author field
as one of the features while training. Unfortunately, from our analysis of the relationship
between the ’author’ and ’category’ fields, we noticed that most authors have written articles
across several categories in our dataset. Hence, there is no pattern here worth pursuing.
2. Preprocessing of data
• We reduced the number of classes in the dataset from 44 to 26. This was done by merging
multiple related classes into one. We had two reasons for doing this:
– Many of the original classes in the dataset were redundant. The dataset had classes like
“science”, “technology” and “science & tech”. Redundant classes like this would have
confused the model.
– Reducing the number of classes in the dataset in this manner increases the number of
records per class. This ensures that each class has more training samples.
• The attributes “headline” and “category” of the dataset were combined into a single attribute
“combined text”.
• Since BERT and RoBERTa are neural networks, they require uniform-length input vectors.
The BERT and RoBERTa tokenizers in the “transformers” library handle the required
padding and truncation during tokenization.
• We train BERT on RoBERTa on both: text that has undergone lemmatization, stopword
removal, and punctuation removal (second phase), as well as text that did not (first phase).
We perform the lemmatization, stopword removal, and punctuation removal using regex
libraries in order to carry out the second phase.
3
3. Feature extraction:
• We removed irrelevant columns to streamline the data set for efficient text classification. For
example, "Author": Typically authorship does not influence the content category, "Link":
Links are external references and usually don’t contribute to text classification and "Date":
The date might not be significant unless the classification is time-sensitive.
• One-Hot Encoding of the “Category” Attribute transforms the categorical "category" attribute
into binary variables. Machine learning models require numerical input, and one-hot
encoding converts categorical data into a binary matrix. Therefore, each of the 26 categories
is represented as a separate column with binary values: 1 for presence and 0 for absence.
4. Training and Validation:
• The distribution of the dataset for this project is organized as follows: the training set
comprises 0.7 of the total data, while both the validation set and the test set each constitute
0.15. To facilitate the training process, we utilized the NCSU VCL (Virtual Computing Lab)
remote desktop, which allowed us to access enhanced GPU capabilities. This strategic use
of advanced hardware significantly reduced our overall training time. In terms of specific
model training durations, the BERT model required approximately 9 hours to complete its
training phase. In comparison, the RoBERTa model demonstrated a slightly more efficient
training process, completing in approximately 6 hours.
• With the above setup we trained four different models:
– BERT trained on text that did not undergo lemmatization, stopword removal, and
punctuation removal
– RoBERTa trained on text that did not undergo lemmatization, stopword removal, and
punctuation removal
– BERT trained on text that underwent lemmatization, stopword removal, and punctuation
removal
– RoBERTa trained on text that underwent lemmatization, stopword removal, and punc-
tuation removal
• Validation was done based on the hyperparameters: number of hidden layers, size of hidden
layers, and dropout rate. For both the BERT models, we decided on 1 dense (output
layer) with activation function ‘softmax’ and 1 dropout layer with a dropout ratio of 0.5
after validation. For both the RoBERTa models, we decided on 2 dense layers, with the
intermediate dense layer having the activation function ‘relu’ and the output dense layer
having the activation function ‘softmax’. The RoBERTa models also had a dropout layer
with a dropout ratio of 0.5.
5. Testing
• The models we obtained were tested using accuracy, precision, recall, and F1-score evalua-
tion metrics (microaverages). The testing results are covered in section 6.
3 Plan and Experiment
Classifying news articles using pre-trained models like BERT (Bidirectional Encoder Representations
from Transformers) or RoBERTa (Robustly optimized BERT approach) involves fine-tuning these
models on a dataset containing labeled news articles.
BERT and RoBERTa are advanced natural language processing models based on transformer archi-
tectures. They are designed to understand the context of words in a sentence by considering the
surrounding words bidirectionally. This bidirectional approach enables them to capture intricate
language patterns and semantics.
3.1 Dataset
The dataset, available on Kaggle, comprises 210,000 records spanning the years 2012 to 2022.
Notably, it exhibits a slight temporal imbalance, with a significant majority of approximately 200,000
4
records predating 2018. This dataset encompasses six attributes: category, headline, author, link, short
description, and date of publication. Among the 210,094 records, these pertain to 42 distinct target
classes (news topics). However, not all of the six attributes are equally pertinent for classification and
analysis. One noteworthy aspect is the presence of around 37,000 records where the ’author’ attribute
contains an empty string value. The step-by-step method to achieve the objective is as follows:
3.2 Hypothesis
We expect that RoBERTa will outperform BERT in terms of accuracy and efficiency for news
classification, given its optimizations over BERT during pre-training.
Removing preprocessing steps like lemmatization, stopword removal, and punctuation removal is
expected improve performance for both BERT and RoBERTa on this news classification task.
Combining the "headline" and "category" attributes into a "combined text" feature will improve
classification accuracy compared to using just the headlines or just the categories.
3.3 Experimental Setup
Libraries:
• Transformers: This library by Hugging Face provides an interface to work with pre-trained
language models, including BERT and RoBERTa.
• PyTorch or TensorFlow: We’ll need a deep learning framework to work with BERT and
RoBERTa. Both PyTorch and TensorFlow are compatible with the Transformers library.
• scikit-learn: This library provides tools for data preprocessing and evaluation metrics. It’s
helpful for preparing the data and evaluating the model’s performance.
• Matplotlib, Seaborn, Pandas, WordCloud:These libraries will be used to plot graphs and
to create insightful visual representations of the data and model performance during the
project.
Algorithms:
• BERT (Bidirectional Encoder Representations from Transformer): It utilizes both directions
in terms of pretraining the model. It can understand the meaning of a word by considering
the words that come before and after it in a sentence.
• RoBERTa (Robustly Optimized BERT Pretraining Approach) Like BERT it also is bidirec-
tional in context of pretraining. Unlike BERT, it removes the next sentence prediction (NSP)
task. RoBERTa has made significant contributions to the field of NLP due to its robust
pretraining approach and improved performance.
Evaluation Metrics:
• F1 Score: The F1-score is the harmonic mean of precision and recall.
• Accuracy: It is calculated as the ratio of correctly predicted instances to the total number of
instances in the dataset.
• Precision: Precision is a measure of the accuracy of a classifier for a specific class
• Recall: Recall, also known as sensitivity or true positive rate, measures the ability of a
classifier to identify all relevant instances of a class.
Hyperparameters:
• Learning Rate:2e-5
• Epoch: 50
• Optimizer: ADAM
• Activation Function: Softmax
Test and Train Split
5
Accuracy Precision Recall F1-
score
BERT model trained without stopword removal, punctua-
tion removal and lemmatization
0.75237 0.75005 0.75237 0.75093
BERT model trained after stopword removal, punctuation
removal and lemmatization
0.73019 0.72662 0.73019 0.72804
RoBERTa model trained without stopword removal, punc-
tuation removal and lemmatization
0.76524 0.75677 0.76524 0.76098
RoBERTa model trained after stopword removal, punctua-
tion removal and lemmatization
0.73226 0.72865 0.73226 0.73045
Table 1: Performance evaluation of the four transformer models we trained)
• 0.7 of the original data is used for training.
• 0.15 of the original data is used for evaluation.
• 0.15 of the original data is used for final testing.
• Random State= 42 (can be any integer) helps to reproduce the same result after reuse of the
train test split.
4 Results
We decided to use accuracy, recall, precision, and F1-score as the evaluation metrics for the models
that we trained. The performance analysis based on these evaluation metrics is presented in Table 1.
From the model performances, we notice the following patterns:
i. The RoBERTa model exhibits better performance than the BERT model for the same
input data despite a shorter training time (RoBERTa completed 4 epochs in about 6 hours
compared to BERT’s 9 hours). In other words, the RoBERTa model trained on text that did
not undergo stopword removal, punctuation removal, and lemmatization does better than the
BERT model trained on the same text that did not undergo stopword removal, punctuation
removal, and lemmatization. The same pattern is observed for the BERT and RoBERTa
models trained on text that was lemmatized and whose stopwords and punctuation were
removed.
ii. Both BERT and RoBERTa appear to be performing better when trained on text that did not
undergo stopword removal, punctuation removal, and lemmatization.
Since all four performance metrics that we took under consideration exhibited the same trend across
the trained models, the above observations could be easily made.
From the confusion matrices of all four models, we noticed that each of the models performs
consistently well in correctly classifying text belonging to all of the classes. In the given confusion
matrix for the BERT model trained on text that did not undergo lemmatization, stopword removal,
and punctuation removal (shown in Figure 1, we can see a darker shade across the diagonal of the
matrix in comparison to the rest of the matrix. This indicates that the number of correct classifications
is much higher than the incorrect classifications for this model. Through the confusion matrix, we
can verify that the favorable results that our model has achieved are due to consistently accurate
classifications across all classes, rather than some classes performing exceptionally well and oth-
ers doing somewhat badly. The confusion matrix looks similar for all four models that we have trained.
The dataset used in this project poses a challenge as many records could belong to multiple news
classes. For instance, a news headline about US politics might fit both "US news" and "politics."
However, the dataset assigns only one class to each headline as the ground truth. Consequently,
our model, though technically accurate, may be deemed incorrect during evaluation if the predicted
6
Figure 1: Confusion Matrix for the BERT model trained on text that did not undergo lemmatization,
stopword removal, and punctuation removal
class differs from the assigned ground truth. This nuance suggests that our model’s performance
could be better than indicated by evaluation results. To enhance future assessments, datasets should
provide multiple correct classes for each record, and evaluation criteria should consider this inherent
ambiguity.
5 Conclusion
The conclusion drawn from this project highlights several key insights into the application of advanced
natural language processing (NLP) techniques for content categorization, specifically within the
domain of news article classification.
1. Model Performance:
• Effectiveness of Transformer Models: The project demonstrates the effectiveness of
transformer-based models, particularly BERT and RoBERTa, in categorizing news headlines.
Both models outperformed traditional machine learning algorithms, such as LSTM and
LinearSVC, by a significant margin. RoBERTa, in particular, showed superior performance
compared to BERT, with higher accuracy and F1-scores, indicating its robustness and
efficiency in processing complex linguistic data.
• Impact of Text Preprocessing: Interestingly, the project findings suggest that preprocessing
steps like lemmatization, stopword, and punctuation removal do not necessarily contribute
to improving model performance for these sophisticated models. Both BERT and RoBERTa
performed better without these preprocessing steps. This could be attributed to the fact
that these models are designed to understand context and semantics more effectively when
provided with complete and unaltered text data.
• Training Efficiency: Despite being trained for the same number of epochs, RoBERTa’s
higher accuracy within a shorter training duration indicates its efficiency. This could be
crucial for scaling up to larger datasets or real-time applications.
2. Data Insights and Challenges:
7
• Class Imbalance and Simplification: The reduction of classes from 44 to 26 was a strategic
decision to address class imbalance, which is a common challenge in real-world datasets.
This simplification likely contributed to more effective training and a more balanced class
representation.
• Feature Extraction and Selection: The decision to focus on certain features (headline and
short description) by removing others like author and date reflects a targeted approach to
feature selection. This likely enhanced the model’s focus on textual content, improving
classification accuracy.
• Category Overlap: The acknowledgment that news headlines might belong to multiple cate-
gories is an important consideration. This overlap presents a challenge for any classification
system, potentially understating the true performance of the models. Future work could
explore multi-label classification approaches to address this.
3. Broader Implications and Future Directions:
• Understanding Language Context: The findings underscore the importance of context in
language understanding. Transformer models’ ability to grasp the nuances of language with-
out oversimplification through preprocessing has implications for various NLP applications
beyond classification, like sentiment analysis and topic modeling.
• Advancements in NLP Models: The success of RoBERTa over BERT in this project adds to
the growing evidence of rapid advancements in NLP models. Continuous improvements in
these models can lead to more sophisticated applications in content analysis, summarization,
and beyond.
• Real-world Applications: These results have practical implications for industries relying
on content categorization, like news aggregation platforms, digital libraries, and content
recommendation systems. The use of such models can significantly enhance the user
experience through better content organization and retrieval.
This project is a compelling demonstration of advanced NLP models tackling real-world challenges,
showcasing cutting-edge technologies’ effectiveness in navigating complexities and revealing insights
in language processing. Beyond immediate applications, its findings shape future NLP research,
providing nuanced insights into language processing and contemporary NLP technologies.In essence,
it serves as a beacon in the evolving NLP landscape, guiding researchers and developers toward
optimized approaches for enhanced solutions.
