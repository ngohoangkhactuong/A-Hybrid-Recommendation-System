from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from surprise import AlgoBase, PredictionImpossible
from sklearn.model_selection import train_test_split
import numpy as np
from MovieLens import MovieLens

class ContentLSTMAlgorithm(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k
        self.max_length = 50
        self.embedding_dim = 64
        self.lstm_units = 32

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        user_descriptions, item_descriptions, labels = self.prepare_data(trainset)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(user_descriptions + item_descriptions)

        user_sequences = tokenizer.texts_to_sequences(user_descriptions)
        item_sequences = tokenizer.texts_to_sequences(item_descriptions)

        user_sequences_padded = pad_sequences(user_sequences, maxlen=self.max_length)
        item_sequences_padded = pad_sequences(item_sequences, maxlen=self.max_length)

        X_train_user, X_test_user, y_train_user, y_test_user = train_test_split(
            user_sequences_padded, np.array(labels), test_size=0.2, random_state=42
        )

        X_train_item, X_test_item, y_train_item, y_test_item = train_test_split(
            item_sequences_padded, np.array(labels), test_size=0.2, random_state=42
        )

        vocab_size = len(tokenizer.word_index) + 1

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=self.embedding_dim, input_length=self.max_length))
        model.add(Bidirectional(LSTM(self.lstm_units, return_sequences=True, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001))))
        model.add(Bidirectional(LSTM(int(self.lstm_units/2), return_sequences=True, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001))))
        model.add(Bidirectional(LSTM(int(self.lstm_units/4), kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001))))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))  # Increased number of neurons
        model.add(Dropout(0.2))  # Added dropout layer
        model.add(Dense(1, activation='linear'))

        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        def lr_schedule(epoch):
            return 0.0001 * (0.1 ** int(epoch / 10))

        scheduler = LearningRateScheduler(lr_schedule)
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Increased patience

        model.fit(X_train_user, y_train_user, epochs=100, batch_size=16, verbose=1,
                  validation_data=(X_test_user, y_test_user), callbacks=[scheduler, early_stopping])

        model.fit(X_train_item, y_train_item, epochs=100, batch_size=16, verbose=1,
                  validation_data=(X_test_item, y_test_item), callbacks=[scheduler, early_stopping])

        self.model = model
        self.tokenizer = tokenizer

        return self

    def prepare_data(self, trainset):
        ml = MovieLens()
        genres = ml.getGenres()
        years = ml.getYears()

        user_descriptions = []
        item_descriptions = []
        labels = []

        for user_id, item_id, rating in trainset.all_ratings():
            user_description = f"User{user_id} likes {genres[item_id]} movies from {years[item_id]}."
            item_description = f"Movie{item_id} is a {genres[item_id]} movie from {years[item_id]}."

            user_descriptions.append(user_description)
            item_descriptions.append(item_description)
            labels.append(rating)

        return user_descriptions, item_descriptions, labels

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item are unknown.')

        user_sequence = self.tokenizer.texts_to_sequences([f"User{u}"])
        item_sequence = self.tokenizer.texts_to_sequences([f"Movie{i}"])

        user_sequence = pad_sequences(user_sequence, maxlen=self.max_length)
        item_sequence = pad_sequences(item_sequence, maxlen=self.max_length)

        predicted_rating_user = self.model.predict([user_sequence])[0][0]
        predicted_rating_item = self.model.predict([item_sequence])[0][0]

        predicted_rating = (predicted_rating_user + predicted_rating_item) / 2.0

        return predicted_rating
