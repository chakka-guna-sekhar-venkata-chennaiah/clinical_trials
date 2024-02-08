import streamlit as st
import time 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import base64
from io import BytesIO
st.header('Clinical Trail')
#patient=5
 
def load_model():
    loaded_model = keras.models.load_model('tuned_model_classic.h5')
    return loaded_model

def value_mapper(df):
    value_mapping = {
                        'no weakness': 0,
                        'mild': 1,
                        'moderate/severe': 2,
                        'normal': 0,
                        'abnormal': 1,
                        'forced deviation': 2,
                    }
    converted_df=df.applymap(lambda x: value_mapping.get(x, 0))
    return converted_df.sum().sum()


def probability_score(score):
    if score <=1: return f'15%'
    elif 2 <= score <=3: return f'30%'
    elif 4<= score <=6: return f'0%-85%'
    return f'0'
    
def get_table_download_link(df):
    excel_writer = BytesIO()
    df.to_excel(excel_writer)
    excel_writer.seek(0)
    b64 = base64.b64encode(excel_writer.read()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="answers.xlsx">Download Answers as Excel Sheet</a>'
    return href
    
if 'patient' not in st.session_state:
    st.session_state.patient=1
patient=st.session_state.patient


q1=st.selectbox('Is the patient on anticoagulant / blood thinner?',['yes', 'no', 'unknown'])
q2=st.selectbox('How old is the patient?',['older than 80 years old', '80 years old or younger', 'age is unknown'])
#q3=st.selectbox('Did anyone see when the symptoms started?',['yes, enter time', 'no'])
#q4=st.selectbox('What time does the patient last seen well?',['yes, enter time', 'unknown'])
q5=st.selectbox('Does the patient have arm weakness?',['no weakness', 'mild', 'moderate / severe'])
user_option=st.selectbox('Are you interested for giving your current facial image?',[' ','yes','no'])
if user_option=='yes':
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read the image and preprocess it
        loaded_model=load_model()
        class_labels=['Stroke','Non-Stroke']
        image = Image.open(uploaded_image)
        image = image.convert('RGB')
        image = image.resize((150, 150))  # Resize to match the model's input shape
        image = np.array(image)  # Convert PIL image to numpy array
        image = image / 255.0  # Normalize pixel values (similar to how you did in the model training)

        # Make prediction using the loaded model
        prediction = loaded_model.predict(np.expand_dims(image, axis=0))[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = prediction[predicted_class_index]
        confidence = float(f'{confidence:.2f}') 
        if confidence>0.5:
            q6='abnormal'
        else:
            q6='normal'
        
        # Display the uploaded image and the prediction
        st.image(image, caption=f'Uploaded Image', use_column_width=True)
        
        # Check if the predicted class is "Non-Stroke" and the confidence is high (you can adjust the threshold)
        if confidence > 0.5:
            st.write(f'Chance of getting stroke')
        else:
            st.write(f'No chance of getting stroke')
   
        
elif user_option=='no':
    q6=st.selectbox('Does the patient have face weakness?',['normal', 'abnormal'])
else:
    pass
q7=st.selectbox('Check speech content & ask the patient to name 3 common items', ['normal', 'abnormal'])
q8=st.selectbox('Ask the patient to show me 2 fingers', ['normal', 'abnormal'])
q9=st.selectbox('Does the patient have gaze deviation to either side?', ['normal', 'gaze preference', 'forced deviation'])
#q10=st.selectbox('Ask the patient are you weak anywhere?', ['normal', 'abnormal'])
#q11=st.selectbox('Ask the patient who this arm belongs to?', ['normal', 'abnormal'])
    

 #   submitted=st.form_submit_button('Submit')
if st.button('Submit'):
    user_tracker={'Is the patient on anticoagulant / blood thinner?':[q1],
                      'How old is the patient?':[q2],
                      #'Did anyone see when the symptoms started?':[q3],
                      #'What time does the patient last seen well?':[q4],
                      'Does the patient have arm weakness?':[q5],
                      'Does the patient have face weakness?':[q6],
                      'Check speech content & ask the patient to name 3 common items':[q7],
                      'Ask the patient to show me 2 fingers':[q8],
                      'Does the patient have gaze deviation to either side?':[q9],
                      #'Ask the patient are you weak anywhere?':[q10],
                      #'Ask the patient who this arm belongs to?':[q11],
                      'Score':[None],
                      'Probability-Percentage':[None]}
    df=pd.DataFrame(user_tracker)
    df=df.T

    df.rename(columns={
            0:f'patient{patient}'
        },inplace=True)
    score=value_mapper(df)
    df.loc['Score']=score
    df.loc['Probability-Percentage']=probability_score(score)
    
    
    st.write(f'Probability of getting heart stroke is {probability_score(score)}')
    finalised_data=pd.read_csv('finalised_data.csv',index_col=['Unnamed: 0'])
    col_name=df.columns
    finalised_data[col_name[0]]=df.iloc[:,0:]
    finalised_data.to_csv('finalised_data.csv')
    st.session_state.patient=st.session_state.patient + 1
    st.info('Data saved successfully!!')
    st.markdown(get_table_download_link(finalised_data), unsafe_allow_html=True)

    