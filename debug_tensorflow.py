# Add this to a new file named debug_tensorflow.py in your repo
import streamlit as st
import os
import sys
import platform

st.title("TensorFlow Debug Info")

# System info
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
        model.compile(optimizer='adam', loss='mse')
        
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
        st.write("Model structure:", loaded_model.summary())
        
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
        st.error("Stack trace:", exc_info=True)