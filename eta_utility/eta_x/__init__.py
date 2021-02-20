import os

# suppress tensorflow cuda  and deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from .eta_x import ETAx, callback_environment
