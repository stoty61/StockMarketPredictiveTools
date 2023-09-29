import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load your historical stock price data as a DataFrame
data = pd.read_csv('stock_data.csv')
# Replace 'stock_data.csv' with your dataset

# Preprocess the data
scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create sequences for training
sequence_length = 10  # Number of previous days to consider
sequences = []
targets = []
for i in range(len(data) - sequence_length):
    sequences.append(data['Close'].values[i:i+sequence_length])
    targets.append(data['Close'].values[i+sequence_length])

# Split data into training and testing sets
X = np.array(sequences)
y = np.array(targets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train an RNN model
rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

rnn_model.compile(optimizer='adam', loss='mean_squared_error')
rnn_model.fit(X_train, y_train, epochs=20, batch_size=32)

# Define and train a Transformer model (you can use libraries like Trax or Tensor2Tensor)
# This example assumes Trax for the Transformer model
import trax
from trax import layers as tl

transformer_model = tl.Serial(
    tl.ShiftRight(mode='train'),  # Shift the input to the right
    tl.EmbeddingAndPositionalEncodings(),  # Add positional encodings to the input 
    # You can add multiple transformer layers here
    tl.Transformer(d_model=32, n_heads=4, d_ff=64),
    tl.Dense(1)
)

# Define a custom loss function for regression tasks
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean((y_true - y_pred) ** 2)

transformer_model.init_from_file("your_pretrained_model.pkl")  # Load a pretrained model if available

trainer = trax.Trainer(
    model=transformer_model,
    loss_fn=mean_squared_error,
    optimizer=trax.optimizers.Adam(0.001),
)

# You may need to define a custom data generator for Trax

# Train the Transformer model
trainer.train(
    train_generator,  # Define your data generator
    steps=1000,  # Define the number of training steps
    eval_steps=100  # Define the number of evaluation steps
)

# Make predictions with both models
rnn_predictions = rnn_model.predict(X_test)
transformer_predictions = transformer_model(X_test)

# Evaluate the models and make trading decisions based on predictions

# You can use standard evaluation metrics and trading strategies for this purpose

# Remember that predicting stock prices is a challenging task, and the above code is a simplified example. Professional trading decisions should involve domain expertise and more advanced models.

