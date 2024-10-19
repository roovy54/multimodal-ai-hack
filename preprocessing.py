import os, pd

directory_path = "public_dataset/train/tweets/"
coins = [f.name for f in os.scandir(directory_path)]
all_tweets = pd.DataFrame()

f = 0
for coin in coins:
    directory_path = f"/content/public_dataset/train/tweets/{coin}"
    dates = [f.name[:-4] for f in os.scandir(directory_path)]
    for date in dates:
        try:
            df = pd.read_csv(
                f"/content/public_dataset/train/tweets/{coin}/{date}.csv",
                on_bad_lines="skip",
            )
        except:
            print("Skipping", coin, date)
            f = 1
        break
    if f == 1:
        continue
        # Remove columns with higher number of NaN values
        # rm_cols = ['place', 'quote_url', 'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'retweet_date', 'translate', 'trans_src', 'trans_dest']
        # for col in rm_cols:
        #   if col not in df.columns:
        #     print('Skipping coin:', coin)
        #     break
        # continue
        # df = df.drop(rm_cols, axis = 1)
        # # Remove useless columns
        # useless_columns = ['id', 'conversation_id', 'created_at', 'user_id', 'username', 'name', 'replies_count', 'reply_to']
        # df = df.drop(useless_columns, axis = 1)

        # sentiment_analysis = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

        # def preprocess_tweet(tweet):
        #   tweet = str(tweet)
        #   # Lowercase
        #   tweet = tweet.lower()
        #   # Remove URLs
        #   tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
        #   # Remove user mentions
        #   tweet = re.sub(r'@\w+', '', tweet)
        #   # Remove hashtags
        #   tweet = re.sub(r'#', '', tweet)
        #   # Remove special characters and punctuation
        #   tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)
        #   # Tokenization
        #   tokens = tweet.split()
        #   # Remove stop words
        #   stop_words = set(stopwords.words('english'))
        #   tokens = [word for word in tokens if word not in stop_words]
        #   # Lemmatization
        #   lemmatizer = WordNetLemmatizer()
        #   tokens = [lemmatizer.lemmatize(word) for word in tokens]
        #   # Rejoin tokens
        #   return ' '.join(tokens)

        # tweets = df['tweet'].apply(preprocess_tweet)
        # transformers_sentiments = tweets.apply(sentiment_analysis)

        # labels = []
        # scores = []
        # for sentiment in transformers_sentiments:
        #   label = sentiment[0]['label']
        #   if label == 'LABEL_0':
        #     label = -1
        #   elif label == 'LABEL_1':
        #     label = 0
        #   elif label == 'LABEL_2':
        #     label = 1
        #   labels.append(label)
        #   scores.append(round(sentiment[0]['score'], 4))

        # df['sentiment_pred'] = labels
        # df['sentiment_score'] = scores
        # Summarize for given date given coin
        def clean_value(value):
            if isinstance(value, str):
                # Remove any whitespace and convert to lowercase
                value = value.strip().lower()

                # Check if the value ends with 'k'
                if value.endswith("k"):
                    return float(value[:-1]) * 1000  # Convert '2k' to 2000.0
                else:
                    try:
                        return float(value)  # Convert to float for other cases
                    except ValueError:
                        return None  # Return None for any non-convertible values
            return float(value)  # For integers and floats, just convert to float

        if "likes_count" in df.columns:
            df["likes_count"] = df["likes_count"].fillna(0)
            df["likes_count"] = df["likes_count"].apply(clean_value)
            total_likes = df["likes_count"].sum()
            df["retweets_count"] = df["retweets_count"].fillna(0)
            df["retweets_count"] = df["retweets_count"].apply(clean_value)
            total_retweets = df["retweets_count"].sum()
        else:
            df["Likes"] = df["Likes"].fillna(0)
            df["Likes"] = df["Likes"].apply(clean_value)
            df["Retweets"] = df["Retweets"].fillna(0)
            df["Retweets"] = df["Retweets"].apply(clean_value)
            total_likes = df["Likes"].apply(int).sum()
            total_retweets = df["Retweets"].sum()

        # sentiment_pred_counts = df['sentiment_pred'].value_counts()
        # total_positives = sentiment_pred_counts.get(1, 0)
        # total_neutrals = sentiment_pred_counts.get(0, 0)
        # total_negatives = sentiment_pred_counts.get(-1, 0)
        data = pd.DataFrame(
            {
                "coin": [coin],  # Use a list
                "date": [date],  # Use a list
                "total_likes": [total_likes],  # Use a list
                "total_retweets": [total_retweets],  # Use a list
                # 'total_positives': [total_positives],  # Use a list
                # 'total_neutrals': [total_neutrals],  # Use a list
                # 'total_negatives': [total_negatives]   # Use a list
            }
        )
        all_tweets = pd.concat([all_tweets, data], ignore_index=True)

# Assuming df is your DataFrame
all_tweets.to_csv("all_tweets.csv", index=False)
