import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from datetime import datetime
from typing import Dict, Text

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("tf is built with cuda: " + str(tf.test.is_built_with_cuda()))


def ask_user_yn(input_str):
    answer = input(input_str)
    while answer not in ["Yes", "yes", "YES", "Y", "y", "No", "no", "NO", "N", "n"]:
        answer = input("Please answer with yes or no!\n")
    return answer


# small or big data (big data might take VERY long)
# dataset source: https://grouplens.org/datasets/movielens/latest/
dataset_choice = input("Choose dataset:\n"
                       "1. Small\n"
                       "2. Big\n")
while dataset_choice not in ["1", "2"]:
    dataset_choice = input("Please answer with 1 (small), or 2 (big)!\n")

if dataset_choice == '2':
    small_data = False
else:
    small_data = True

if small_data:
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    ratings_df = pd.read_csv('data/ml-latest-small/ratings.csv')
else:
    movies = pd.read_csv("data/ml-latest/movies.csv")
    ratings_df = pd.read_csv('data/ml-latest/ratings.csv')

pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

#   DATA PROCESSING

# converting timestamps to date
ratings_df['date'] = ratings_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
ratings_df.drop('timestamp', axis=1, inplace=True)

# merging movie information with ratings data
ratings_df = ratings_df.merge(movies[['movieId', 'title', 'genres']], left_on='movieId',
                              right_on='movieId', how='left')

# needs to be string for StringLookup later
ratings_df['userId'] = ratings_df['userId'].astype(str)

# selecting the features we want to use and transform it into TensorFlow datasets
ratings = tf.data.Dataset.from_tensor_slices(dict(ratings_df[['userId', 'title', 'rating']]))
movies = tf.data.Dataset.from_tensor_slices(dict(movies[['title']]))

# maps to desired format
ratings = ratings.map(lambda x: {
    "title": x["title"],
    "userId": x["userId"],
    "rating": float(x["rating"])
})
movies = movies.map(lambda x: x["title"])

# data info
total_data = len(ratings)
print('Total Data: {}'.format(total_data))

# calculating 80/20 split for training and evaluation set
small_subset = total_data // 5
big_subset = total_data - small_subset

# setting random seed to 42 ofc
tf.random.set_seed(42)
shuffled = ratings.shuffle(total_data, seed=42, reshuffle_each_iteration=False)

# separating data into training and evaluation set
train = shuffled.take(big_subset)
test = shuffled.skip(big_subset).take(small_subset)

# getting unique user ids and movie titles
# Needed for when we map the raw values of our categorical features to embedding vectors in the models.
# To do that, we need a vocabulary that maps a raw feature value to an integer in a contiguous range, allowing
# us to look up the corresponding embeddings in our embedding tables.
movie_titles = movies.batch(1_000_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["userId"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

print('Unique Movies: {}'.format(len(unique_movie_titles)))
print('Unique users: {}'.format(len(unique_user_ids)))


class MovieModel(tfrs.models.Model):
    def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
        # we take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()

        # dimensionality of the embeddings for models
        # in this context embedding_dimension = 32 means that each movie and user is represented as
        # a vector in 32-dimensional space.
        embedding_dimension = 32

        # we use a Keras preprocessing layer to convert user ids to integers, and then convert those to embeddings
        # using the Embedding layer. The unique_user_ids are then used as vocabulary
        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # Same as above, Keras preprocessing to convert ids to integers, and then convert those
        # to embeddings using the Embedding layer. The unique_movie_titles are then supplied as vocabulary
        self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

        # rating model that takes user and movie embeddings to predict rating of a movie by that user
        # Can be made arbitrarily complicated as long as output is a scalar
        # relu = rectified linear unit activation function.
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        # The tasks

        # ranking task
        self.ranking_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

        # retrieval task
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.movie_model)
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]):
        # take user features and pass them into the user model.
        user_embeddings = self.user_model(features["userId"])
        # take movie features and pass them into the movie model.
        movie_embeddings = self.movie_model(features["title"])

        return user_embeddings, \
               movie_embeddings, self.rating_model(tf.concat([user_embeddings, movie_embeddings], axis=1))


    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        ratings = features.pop("rating")

        user_embeddings, movie_embeddings, rating_predictions = self(features)

        # compute the loss for each task.
        rating_loss = self.ranking_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

        # combine loss using the loss weights.
        return (self.rating_weight * rating_loss
                + self.retrieval_weight * retrieval_loss)


load_answer = ask_user_yn("\nLoad previously saved model?: Y/N\n")

if load_answer in ["Yes", "yes", "YES", "Y", "y"]:
    model = tf.saved_model.load("export")
else:
    print("Not loading previous model")
    # instantiating model with given loss weights
    model = MovieModel(rating_weight=1, retrieval_weight=1)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    # shuffle, batch and caching train+eval data
    cached_train = train.shuffle(total_data).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    # train model
    model.fit(cached_train, epochs=3, verbose=2)

    # evaluate model on test set
    metrics = model.evaluate(cached_test, return_dict=True, verbose=2)
    print(f"\nRetrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}")
    print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}")


def predict_movie(user, top_n=3):
    # Creating a model that takes in raw query features using and recommends movies using BruteForce layer
    # (can be sped up by approximating predictions using ScaNN, but not necessary for "small" candidate sets)
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    index.index_from_dataset(
        tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
    )

    # collect movies already seen, to exclude from recommendation
    watched = ratings_df[ratings_df['userId'] == str(user)]
    watched = watched['title'].to_numpy()

    # get top top_n recommendations for user (excluding already watched movies)
    _, titles = index.query_with_exclusions(
        queries=tf.constant([str(user)])
        , exclusions=np.array([watched]), k=top_n
    )

    # Get recommendations not excluding already seen
    # _, titles = index(tf.constant([str(user)]))

    print('\nTop {} recommendations for user {}:\n'.format(top_n, user))
    for i, title in enumerate(titles[0, :top_n].numpy()):
        print('{}. {}'.format(i + 1, title.decode("utf-8")))


def predict_rating(user, movie):
    trained_movie_embeddings, trained_user_embeddings, predicted_rating = model({
        "userId": np.array([str(user)]),
        "title": np.array([movie])
    })
    print("\nRating prediction for the movie {}: {}".format(movie, predicted_rating.numpy()[0][0]))


# input section
user_id_or_no = input("Enter a user id between 1 and " + str(len(unique_user_ids)) + " or No to exit: ")
while user_id_or_no not in ["No", "no", "NO", "N", "n"]:
    if bytes(user_id_or_no, 'utf-8') in unique_user_ids:
        action_input = input("\nChoose action:\n"
                             "1. See top 10 recommended movies for the user\n"
                             "2. See the predicted rating for a given movie and the user\n"
                             "3. See user rating history\n"
                             "4. Change user\n"
                             "5. Exit\n")

        while action_input != "5":
            if action_input not in ["1", "2", "3", "4", "5"]:
                action_input = input("Invalid input, please try again:\n"
                                     "1. See top 10 recommended movies for the user\n"
                                     "2. See the predicted rating for a given movie and the user\n"
                                     "3. See user rating history\n"
                                     "4. Change user\n"
                                     "5. Exit")
            else:
                if action_input == '1':
                    predict_movie(user_id_or_no, 10)
                elif action_input == '2':
                    input_movie_title = input("\nInput movie title or no to exit: ")
                    while bytes(input_movie_title, 'utf-8') not in unique_movie_titles and \
                            input_movie_title not in ["No", "no", "NO", "N", "n"]:
                        input_movie_title = input(
                            input_movie_title + " not found, remember to add the year of the movie,"
                                                " e.g. Minions (2015)\n")
                    if input_movie_title not in ["No", "no", "NO", "N", "n"]:
                        predict_rating(user_id_or_no, input_movie_title)
                elif action_input == '3':
                    print('\nRating history for user {} :\n'.format(user_id_or_no))
                    print(ratings_df[ratings_df['userId'] == user_id_or_no])
                elif action_input == '4':
                    user_id_or_no = input("Input user id: ")
                elif action_input == '5':
                    user_id_or_no = 'n'
            if action_input != '5':
                action_input = input("\n1. See top 10 recommended movies for the user\n"
                                     "2. See the predicted rating for a given movie and the user\n"
                                     "3. See user rating history\n"
                                     "4. Change user\n"
                                     "5. Exit\n")
        if action_input == '5':
            user_id_or_no = 'n'
    else:
        user_id_or_no = input("Invalid input\nPlease enter a user id between 1 and " +
                              str(len(unique_user_ids)) + " or No to exit\n")

save_answer = ask_user_yn("Save model?: Y/N\n")

if save_answer in ["Yes", "yes", "YES", "Y", "y"]:
    model.retrieval_task = tfrs.tasks.Retrieval()
    # model.compile()
    tf.saved_model.save(model, "export")
else:
    print("Not saving")

print("Done")
