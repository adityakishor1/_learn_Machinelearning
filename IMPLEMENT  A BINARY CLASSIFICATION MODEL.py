import pandas as pd
import numpy as np
import tensorflow as tf

# Load data
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

# Shuffle the training set
train_df = train_df.reindex(np.random.permutation(train_df.index))

# Normalize data
train_df_norm = (train_df - train_df.mean()) / train_df.std()
test_df_norm = (test_df - test_df.mean()) / test_df.std()

# Create binary target column
threshold = 265000  # This is the 75th percentile for median house values.
train_df_norm["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(int)
test_df_norm["median_house_value_is_high"] = (test_df["median_house_value"] > threshold).astype(int)

# Print out a few example cells from the beginning and middle of the training set
print(train_df_norm["median_house_value_is_high"].head())
print(train_df_norm["median_house_value_is_high"].iloc[4000:4005])

# Define inputs
inputs = {
    'median_income': tf.keras.Input(shape=(1,)),
    'total_rooms': tf.keras.Input(shape=(1,))
}

# Hyperparameters
learning_rate = 0.001
epochs = 20
batch_size = 100
classification_threshold = 0.35
label_name = "median_house_value_is_high"

# Metrics
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=classification_threshold),
    tf.keras.metrics.Precision(thresholds=classification_threshold, name='precision'),
    tf.keras.metrics.Recall(thresholds=classification_threshold, name='recall')
]

# Model creation function
def create_model(inputs, learning_rate, METRICS):
    input_layers = {name: tf.keras.layers.Input(shape=(1,), name=name) for name in inputs}
    concatenated_inputs = tf.keras.layers.Concatenate()(list(input_layers.values()))
    x = tf.keras.layers.Dense(128, activation='relu')(concatenated_inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_layers, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=METRICS)
    return model

# Training function
def train_model(model, dataset, epochs, label_name, batch_size):
    features = {name: np.array(value) for name, value in dataset.items() if name != label_name}
    label = np.array(dataset[label_name])
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist

# Plotting function
import matplotlib.pyplot as plt

def plot_curve(epochs, hist, list_of_metrics_to_plot):
    plt.figure()
    for metric in list_of_metrics_to_plot:
        plt.plot(epochs, hist[metric], label=metric)
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Establish the model's topography
my_model = create_model(inputs, learning_rate, METRICS)

# Train the model on the training set
epochs, hist = train_model(my_model, train_df_norm, epochs, label_name, batch_size)

# Plot metrics vs. epochs
list_of_metrics_to_plot = ['accuracy', 'precision', 'recall']
plot_curve(epochs, hist, list_of_metrics_to_plot)
