from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam

def model_builder(input_shape, lstm_units=50, dropout_rate=0.2, dense_units=25, learning_rate=0.001):
    """
    Crée et compile un modèle LSTM pour la prédiction.

    Args:
        input_shape (tuple): La forme des données d'entrée (sequence_length, n_features).
        lstm_units (int): Nombre d'unités LSTM dans chaque couche.
        dropout_rate (float): Taux de Dropout pour régulariser le modèle.
        dense_units (int): Nombre de neurones dans la couche Dense intermédiaire.
        learning_rate (float): Taux d'apprentissage pour l'optimiseur Adam.

    Returns:
        model (tf.keras.Model): Modèle LSTM compilé.
    """
    model = Sequential([
        # Première couche LSTM
        LSTM(lstm_units, activation='relu', return_sequences=True, kernel_regularizer=l2(learning_rate), input_shape=input_shape),
        Dropout(dropout_rate),
        
        # Deuxième couche LSTM
        LSTM(lstm_units, activation='relu', return_sequences=True, kernel_regularizer=l2(learning_rate)),
        Dropout(dropout_rate),
        
        # Troisième couche LSTM
        LSTM(lstm_units, activation='tanh', return_sequences=False, kernel_regularizer=l2(learning_rate)),
        Dropout(dropout_rate),
        
        # Couche Dense intermédiaire
        Dense(dense_units, activation='relu'),
        # Couche de sortie
        Dense(1)
    ])

    # Compilation du modèle
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model

def RSI(series, period=10):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))