# This file is a diagnostic tool, it enables us to check if our system is ready to run artificial intelligence models using TensorFlow.
# It also tests if a small model can be created, saved, loaded, and used to make a prediction.
# We had to create this file because Arnaud Alves had some issues with the artificial intelligence model, Claude AI suggested to creat this file.

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
st.write(f"Python version: {sys.version}") # Show the version of Python currently being used
st.write(f"Platform: {platform.platform()}") # Show the name and version of the operating system (e.g. Windows, Mac, Linux)
st.write(f"Architecture: {platform.architecture()}") # Show whether the system is 32-bit or 64-bit
st.write(f"Working directory: {os.getcwd()}") # Show the current folder where this program is running

# Check if TensorFlow is installed and working
st.subheader("TensorFlow Information")
try:
    import tensorflow as tf # Try to import TensorFlow
    st.write(f"TensorFlow version: {tf.__version__}") # Show which version of TensorFlow is installed
    st.write(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")  # Show whether TensorFlow was installed with CUDA (for GPU acceleration)
    st.write(f"TensorFlow backend: {tf.config.list_physical_devices()}")  # Show the hardware (CPU or GPU) that TensorFlow can use
    st.success("✅ TensorFlow is installed and working")     # If everything worked, display a success message
except ImportError:
    st.error("❌ TensorFlow is not installed") # This happens if TensorFlow is not installed at all
except Exception as e:
    st.error(f"❌ Error importing TensorFlow: {str(e)}") # If there was another kind of error, show it here

# Try creating and testing a very simple TensorFlow model
st.subheader("Model Loading Test")

# Create a button on the web page — it must be clicked to run the test
test_button = st.button("Test Loading a Simple Model")
if test_button:
    try:
        # Import TensorFlow again (just to be safe), and NumPy to generate fake data
        import tensorflow as tf
        import numpy as np
        
        st.write("Creating a simple test model...")
        
        # This builds a very simple artificial neural network with 2 layers
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, input_shape=(3,)), # first layer takes 3 numbers as input
            tf.keras.layers.Dense(1) # second layer outputs a single number
        ])
        # Prepare the model for training using a common algorithm and error measurement
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Generate 10 random "fake" examples to train the model
        X = np.random.rand(10, 3) # 10 rows of 3 input numbers
        y = np.random.rand(10, 1)  # 10 target outputs (1 number each)
        # Train the model with this fake data, just for 1 round
        model.fit(X, y, epochs=1, verbose=0)
        
        # Save the model to a file so it can be reused later
        test_model_path = "test_model.h5"
        model.save(test_model_path)
        st.write(f"Model saved to {test_model_path}")
        
        # Now try loading the model from the file
        st.write("Loading the test model...")
        loaded_model = tf.keras.models.load_model(test_model_path)
        st.write("Model loaded successfully!")
        
        # Make a test prediction using a new fake input
        test_input = np.random.rand(1, 3) # 1 input with 3 features
        prediction = loaded_model.predict(test_input)
        st.write(f"Test prediction: {prediction}")
        
        st.success("✅ Test model creation, saving, and loading works!")
        
        # Clean up by deleting the saved test model file
        os.remove(test_model_path)
        st.write("Test model file removed")

    # If anything goes wrong during the test, show the error
    except Exception as e:
        st.error(f"❌ Test model loading failed: {str(e)}")
        st.error("Check the logs for more details")
