To extract data, we employ files with the.json extension. The features of the dataset are as follows: The column that we will use most frequently for our work is the "text" column. To determine the severity of the depression, use this column. We use 19 of these 29 columns for additional computation. We get 50 distinct user IDs after normalizing the user id data. The 'text' column is still dirty, though. Four null values were found in the "text" column after calculation. Hence, before continuing, we substitute empty strings for these null values. We do not currently have any null values. The next step is to organize the tweets by user ID. We accomplish this using Python's group by() function. We obtain a dataframe after grouping that contains user IDs and the concatenated tweets.

The quantity of tweets that were sent from the associated account ID must also be added. The command that is issued in addition to the previous one in order to complete this task is as follows: We save this number in a column called "count."
We must now tidy up this additional text. In order to do this, we make use of the WordNetLemmatizer() function from the NLTK package. In order to gather the tokens in the text, we additionally tokenize the clean text. These are kept in two new columns we've created, clean text and clean tokens.

With the help of a dense semi-supervised labelling approach, we have now labelled the data. This dataframe will be used to determine the depression score. We make use of the Gensim library to accomplish this. The 'punkt' and 'vader lexicon' libraries from the NLTK library must first be imported. The intensity is then defined as being larger than 0.5 or less than 0.5 in wide terms and narrow terms, respectively.

Our final task is to calculate the polarity, semantic, and depression scores of each user ID. We perform this by iterating through the dataframe and using the SentimentIntensityAnalyzer() function. We calculate final depression score using the following formula: 

![image](https://github.com/user-attachments/assets/50a27ae9-cc87-4978-a128-d6d7b32f3038)


However, with respect to the formula we need to normalize the depression score such that the values lie between [0, 1]. We can do this by using MinMaxScaler() from Scikit-Learn library. A depression score closer to 1 shows that the user is severely depressed whereas a value close to 1 show that the user is not depressed. We can use this dataframe for further analysis and prediction, preferably by using LSTMs and other deep learning techniques.

After creating a dataframe with clean text, tokens, count of tweets by the corresponding user, polarity score, semantic score, and depression score, we create unigrams, bigrams and trigrams which have the following words and icons:
 
![image](https://github.com/user-attachments/assets/1db18b47-4324-4146-8691-85edbd7a6440)

We also store the count of these n-grams for the corresponding user and store these n-grams in different lists for further processing. The we create three models, reach for unigrams, bigrams, and trigrams using the gensim.models.Word2Vec() function and passing the corresponding lists we created in the previous step as parameters. We filter the 150 w2v embeddings on the basis of keys specified above and store them in the dataframe. For further prediction of the depression score using an LSTM model, we use these 150 w2v embeddings.

![image](https://github.com/user-attachments/assets/ff6fe011-1aff-4f5e-8be8-47628b15c70c)

OVERALL PIPELINE OF THE PROPOSED MODEL:

![image](https://github.com/user-attachments/assets/d9390379-ba60-4713-9abb-b15e44406f32)

Performance Analysis:
![image](https://github.com/user-attachments/assets/33f95346-e033-4f29-8802-71358f820b54)

We take into consideration the common practices and abbreviations that are followed during texting and try to replace it with words which would be helpful in performing the required task. This is done with the help of:

![image](https://github.com/user-attachments/assets/3a5f8b1e-a045-4588-8c3b-b2072b04acb1)


The model summary is as follows:
![image](https://github.com/user-attachments/assets/7b16d0f9-7873-413e-aa23-0524a07e3ce3)

The input provided to the model has the shape of (150, 1). This is because the w2v embeddings received for each trigram containing the keywords is of the length 150. We use these w2v embeddings as an input to our LSTM model. The results obtained after using Adam optimizer with a learning rate of 0.01 with ‘linear’ as the activation function is as follows:
![image](https://github.com/user-attachments/assets/1ce6e3ad-1068-46ca-aa4f-473c53fc3937)

For comparison, we create another sequential LSTM model. However, instead of using three LSTM layers, we use one LSTM layer with a dropout layer where only 80% values are passed on to following layers. The final output is received from a Dense layer with 1 unit and we use Adam as the optimizer with a learning rate of 0.01 and the activation function is softmax since we need to get a probabilistic value between the range of [0, 1] which is our depression score.The model summary is as follows:
![image](https://github.com/user-attachments/assets/7b9b3c09-8ca0-4bf3-ae65-67897753c0ef)

The results obtained for this model is similar to that of the previous one with the MSLE plot being as follows:
![image](https://github.com/user-attachments/assets/0fbf6cef-fe6f-4dca-bcb7-bf065018bcf8)

Finally, we use the first model with a decreased learning rate of 0.0001. We get the best results using a combination of the parameters of the first model and a learning rate of 0.0001.The MSLE plot obtained after 100 epochs is:

![image](https://github.com/user-attachments/assets/7bd0b91b-a9ff-4637-84c7-f82c6b03e400)
After 5-fold cross validation, the average MSLE obtained for our model is around 1.43. We try various combinations of features for prediction of the depression intensity score. The tabulation of the results obtained is as follows:
![image](https://github.com/user-attachments/assets/6ba5a87a-0dbe-42a4-8520-8e580c5b7633)

![image](https://github.com/user-attachments/assets/d8dabfd8-9ed8-4f43-bfb1-0f0b8f35b645)

We can see that the best results of all the models chosen for comparison is shown by our model. In future, we can fine-tune the hyperparameters so that we can obtain better results and also perform classification tasks instead of continuous value prediction.
