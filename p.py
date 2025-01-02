# Test imports
import tensorflow as tf
from tensorflow import keras
print(f"TensorFlow version: {tf.__version__}")

# For the specific layers we need
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential