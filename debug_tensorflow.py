# This file is a diagnostic tool, it enables us to check if our system is ready to run artificial intelligence models using TensorFlow.
# It also tests if a small model can be created, saved, loaded, and used to make a prediction.
# We had to create this file because Arnaud Alves had some issues with the artificial intelligence model.

# We import streamlit, the tool we use to build our webapp
import streamlit as st

# We import tools to get the current information about our system
import os # lets us work with files and folders
import sys # gives access to system-level details like Python version
import platform # helps get operating system information

# This sets the main title at the top of the web page
st.title("TensorFlow Debug Info")

# Shows basic information about the computer running the app
st.subheader("System Information")
st.write(f"Python version: {sys.version}")
st.write(f"Platform: {platform.platform()}")
st.write(f"Architecture: {platform.architecture()}")
st.write(f"Working directory: {os.getcwd()}")

# TensorFlow info
st.subheader("TensorFlow Information")
try:
    import tensorflow as tf
    st.write(f"TensorFlow version: {tf.__version__}")
    st.write(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    st.write(f"TensorFlow backend: {tf.config.list_physical_devices()}")
    st.success("✅ TensorFlow is installed and working")
except ImportError:
    st.error("❌ TensorFlow is not installed")
except Exception as e:
    st.error(f"❌ Error importing TensorFlow: {str(e)}")

# Test model loading capabilities
st.subheader("Model Loading Test")

test_button = st.button("Test Loading a Simple Model")
if test_button:
    try:
        # Create and save a simple test model
        import tensorflow as tf
        import numpy as np
        
        st.write("Creating a simple test model...")
        
        # Create a very simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, input_shape=(3,)),
            tf.keras.layers.Dense(1)
        ])
        # Use a different loss function that's less likely to cause errors
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Fit with some dummy data
        X = np.random.rand(10, 3)
        y = np.random.rand(10, 1)
        model.fit(X, y, epochs=1, verbose=0)
        
        # Save the model
        test_model_path = "test_model.h5"
        model.save(test_model_path)
        st.write(f"Model saved to {test_model_path}")
        
        # Load the model back
        st.write("Loading the test model...")
        loaded_model = tf.keras.models.load_model(test_model_path)
        st.write("Model loaded successfully!")
        
        # Test prediction
        test_input = np.random.rand(1, 3)
        prediction = loaded_model.predict(test_input)
        st.write(f"Test prediction: {prediction}")
        
        st.success("✅ Test model creation, saving, and loading works!")
        
        # Clean up
        os.remove(test_model_path)
        st.write("Test model file removed")
        
    except Exception as e:
        st.error(f"❌ Test model loading failed: {str(e)}")
        st.error("Check the logs for more details")
