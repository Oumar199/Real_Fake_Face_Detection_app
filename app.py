from transformers import ViTForImageClassification, ViTFeatureExtractor
from fake_face_detection.metrics.make_predictions import get_attention
from torchvision import transforms
import streamlit as st
from PIL import Image
import numpy as np
import pickle
import torch
import cv2

# set the color of the header
def header(text):
    st.markdown(f"<h1 style = 'color: #4B4453; text-align: center'>{text}</h1>", unsafe_allow_html=True)
    st.markdown("""---""")

# initialize the size
size = (224, 224)

# let us add a header
header("FAKE AND REAL FACE DETECTION")

# let us add an expander to write some description of the application
expander = st.expander('Description', expanded=True)

with expander:
    st.write('''This website aims to help internet users
            know if a profile is safe by verifying if its displayed face is verifiable. You can download the image
             of a person on Facebook, Whatsapp, or any other social media
             and add it here and click on the submit button to obtain
             the result (fake or actual). You will also receive a
             modification of the original image indicating which
             part of it is suspect or make the site identify if the
             picture is accurate. Enjoy!''')

# let us initialize two columns
left, mid, right = st.columns(3)

# the following function will load the model (must be in cache)
@st.cache_resource
def get_model():
    
    # let us load the image characteristics
    with open('data/extractions/fake_real_dict.txt', 'rb') as f:
        
        depick = pickle.Unpickler(f)
        
        characs = depick.load()
    
    # define the model name
    model_name = 'google/vit-base-patch16-224-in21k'
    
    # recuperate the model
    model = ViTForImageClassification.from_pretrained(
        'data/checkpoints/model_lhGqMDq/checkpoint-440',
        num_labels = len(characs['ids']),
        id2label = {name: key for key, name in characs['ids'].items()},
        label2id = characs['ids']
    )
    
    # recuperate the feature_extractor
    feature_extractor = ViTFeatureExtractor(model_name)
    
    return model, feature_extractor

# let us add a file uploader
st.subheader("Choose an image to inspect")
file = st.file_uploader("", type='jpg')

# if the file is correctly uploaded make the next processes
if file is not None:
    
    # convert the file to an opencv image
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    # resize the image
    opencv_image = cv2.resize(opencv_image, size)
    
    # Let us display the image
    left.header("Loaded image")
    
    left.image(opencv_image, channels='BGR')
    
    left.markdown("""---""")
    
    # initiliaze the smoothing parameters
    smooth_scale = st.sidebar.slider("Smooth scale", min_value=0.1, max_value =1.0, step = 0.1)
    
    smooth_size = st.sidebar.slider("Smooth size", min_value=1, max_value =10)
    
    smooth_iter = st.sidebar.slider("Smooth iter", min_value=1, max_value =10)
    
    # add a side for the scaler and the head number
    scale = st.sidebar.slider("Attention scale", min_value=30, max_value =200)

    head = int(st.sidebar.selectbox("Attention head", options=list(range(1, 13))))

    
    if left.button("SUBMIT"):
        
        # Let us convert the image format to 'RGB'
        image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        # Let us convert from opencv image to pil image
        image = Image.fromarray(image)
        
        with torch.no_grad():
            
            # Recuperate the model and the feature extractor
            model, feature_extractor = get_model()
            
            # Change to evaluation mode
            _ = model.eval()
            
            # Apply transformation on the image
            image_ = feature_extractor(image, return_tensors = 'pt')
            
            # # Recuperate output from the model
            outputs = model(image_['pixel_values'], output_attentions = True)
            
            # Recuperate the predictions
            predictions = torch.argmax(outputs.logits, axis = -1)
            
            # Write the prediction to the middle
            mid.markdown(f"<h2 style='text-align: center; padding: 2cm; color: black; background-color: orange; border: darkorange solid 0.3px; box-shadow: 0.2px 0.2px 0.6px 0.1px gray'>{model.config.id2label[predictions[0].item()]}</h2>", unsafe_allow_html=True)
            
            # Let us recuperate the attention
            attention = outputs.attentions[-1][0]
            
            # Let us recuperate the attention image
            attention_image = get_attention(image, attention, size = (224, 224), patch_size = (14, 14), scale = scale, head = head, smooth_scale = smooth_scale, smooth_size = smooth_size, smooth_iter = smooth_iter)

            # Let us transform the attention image to a opencv image
            attention_image = cv2.cvtColor(attention_image.astype('float32'), cv2.COLOR_RGB2BGR)
            
            # Let us display the attention image
            right.header("Attention")
            
            right.image(attention_image, channels='BGR')
            
            right.markdown("""---""")
