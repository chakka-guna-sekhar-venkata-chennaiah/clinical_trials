import streamlit as st
import time 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import base64
from io import BytesIO
#st.header('Clinical Trail')
#patient=5
 
def load_model():
    loaded_model = keras.models.load_model('tuned_model_classic.h5')
    return loaded_model

def value_mapper(df):
    value_mapping = {
                        'কোনো দুর্বলতা নেই': 0,
                        'মাঝারি দুর্বলতা': 1,
                        'প্রচন্ড দুর্বলতা': 2,
                        'স্বাভাবিক': 0,
                        'অস্বাভাবিক': 1,
                        'কিছুটা অস্বাভাবিক':1,
                        'বেশি অস্বাভাবিক': 2,
                    }
    converted_df=df.applymap(lambda x: value_mapping.get(x, 0))
    return converted_df.sum().sum()


def probability_score(score):
    if score <=1: return f'<15%'
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


#q1=st.selectbox('Is the patient on anticoagulant / blood thinner?',['yes', 'no', 'unknown'])
#q2=st.selectbox('How old is the patient?',['older than 80 years old', '80 years old or younger', 'age is unknown'])
#q3=st.selectbox('Did anyone see when the symptoms started?',['yes, enter time', 'no'])
#q4=st.selectbox('What time does the patient last seen well?',['yes, enter time', 'unknown'])

q5=st.selectbox('রোগীর হাত নাড়াতে কোনো দুর্বলতা আছে কি?',['কোনো দুর্বলতা নেই', 'মাঝারি দুর্বলতা', 'প্রচন্ড দুর্বলতা'])
q5_descrpt='''
    <div>
    অপশনের বর্ণনা:-
    </div>
    <div>
    🔸 কোনো দুর্বলতা নেই : দুই হাত ১০ সেকেন্ডের বেশি তুলে রাখতে পারেন এবং নামাতে পারেন 
    </div>
    <div>
    🔸 মাঝারি দুর্বলতা : একটি হাত তুলতে পারেন কিন্তু ১০ সেকেন্ডের বেশি তুলে রাখতে পারেন না  
    </div>
    🔸 রচন্ড দুর্বলতা : একটি অথবা দুইটি হাত নিজে নিজে একদমই তুলতে পারছেন না 
    '''
st.markdown(q5_descrpt,unsafe_allow_html=True)
st.write(' ')
user_option=st.selectbox('আপনি কি আপনার বর্তমান মুখের ছবি দিতে আগ্রহী?',[' ','হ্যাঁ','না'])
if user_option=='হ্যাঁ':
    uploaded_image = st.file_uploader("একটি ছবি চয়ন করুন...", type=["jpg", "jpeg", "png"])

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
            q6='অস্বাভাবিক'
        else:
            q6='স্বাভাবিক'
        
        # Display the uploaded image and the prediction
        st.image(image, caption=f'চিত্র আপলোড', use_column_width=True)
        
        # Check if the predicted class is "Non-Stroke" and the confidence is high (you can adjust the threshold)
        if confidence > 0.5:
            st.write(f'স্ট্রোক হওয়ার সম্ভাবনা')
        else:
            st.write(f'স্ট্রোক হওয়ার সম্ভাবনা নেই')
   
        
elif user_option=='না':
    q6_descrpt=''' 
    <div>
    অপশনের বর্ণনা:-
    </div>
    <div>
    🔸 স্বাভাবিক  : রোগীর চেহারার দুই পাশ স্বাভাবিক 
    </div>
    🔸 অস্বাভাবিক  : রোগীর চেহারার এক পাশ বেঁকে গেছে
    '''
    
    st.write(' ')
    q6=st.selectbox('রোগীর মুখের দুর্বলতা আছে কি?',['স্বাভাবিক', 'অস্বাভাবিক'])
    st.markdown(q6_descrpt,unsafe_allow_html=True)
    st.write(' ')
else:
    pass
st.write(' ')
q7=st.selectbox('রোগী কথা বলতে পারছে কি এবং রোগীকে ৩ টি সাধারণ জিনিসের নাম বলতে বলুন', ['স্বাভাবিক', 'অস্বাভাবিক'])
q7_descrpt='''
    <div>
    অপশনের বর্ণনা:-
    </div>
    <div>
    🔸 স্বাভাবিক : রোগী স্বাভাবিক ভাবে ৩ টি জিনিসের নাম বলতে পারছে। 
    </div>
    🔸 অস্বাভাবিক : অস্বাভাবিক এবং ০-১ টি নাম ঠিক ভাবে বলছেন
    '''
st.markdown(q7_descrpt,unsafe_allow_html=True)
st.write(' ')

q8=st.selectbox('রোগীকে ২ টি আঙুল দেখাতে বলুন', ['স্বাভাবিক', 'অস্বাভাবিক'])
q8_descrpt='''
    <div>
    অপশনের বর্ণনা:-
    </div>
    <div>
    🔸 স্বাভাবিক : রোগী স্বাভাবিক ভাবে ২ টি আঙুল দেখাতে পারছে।  
    </div>
    🔸অস্বাভাবিক : অস্বাভাবিক এবং ০-১ টি নাম ঠিক ভাবে বলছেন
     '''
st.markdown(q8_descrpt,unsafe_allow_html=True)
st.write(' ')
q9=st.selectbox('রোগীর চোখ একদিকে বেঁকে গেছে কি?', ['স্বাভাবিক', 'কিছুটা অস্বাভাবিক', 'বেশি অস্বাভাবিক'])
q9_descrpt='''
    <div>
    অপশনের বর্ণনা:-
    </div>
    <div>
    🔸 স্বাভাবিক : একটি আঙুল দুই চোখের সামনে নাড়ালে চোখের মণি তা অনুসরণ করতে পারে 
    </div>
    <div>
    🔸 কিছুটা অস্বাভাবিক : একটি আঙুল দুই চোখের সামনে নাড়ালে চোখের মণি তা অনুসরণ করতে কষ্ট হচ্ছে 
    </div>
    🔸 বেশি অস্বাভাবিক : একটি আঙুল দুই চোখের সামনে নাড়ালে চোখের মণি তা অনুসরণ করতে পারছে না
    '''
st.markdown(q9_descrpt,unsafe_allow_html=True)
#q10=st.selectbox('Ask the patient are you weak anywhere?', ['normal', 'abnormal'])
#q11=st.selectbox('Ask the patient who this arm belongs to?', ['normal', 'abnormal'])
    

 #   submitted=st.form_submit_button('Submit')
st.write(' ')
if st.button('জমা'):
    user_tracker={
                      #'Is the patient on anticoagulant / blood thinner?':[q1],
                      #'How old is the patient?':[q2],
                      #'Did anyone see when the symptoms started?':[q3],
                      #'What time does the patient last seen well?':[q4],

                      'রোগীর হাত নাড়াতে কোনো দুর্বলতা আছে কি?':[q5],
                      'রোগীর মুখের দুর্বলতা আছে কি?':[q6],
                      'রোগী কথা বলতে পারছে কি এবং রোগীকে ৩ টি সাধারণ জিনিসের নাম বলতে বলুন':[q7],
                      'রোগীকে ২ টি আঙুল দেখাতে বলুন':[q8],
                      'রোগীর চোখ একদিকে বেঁকে গেছে কি?':[q9],
                      #'Ask the patient are you weak anywhere?':[q10],
                      #'Ask the patient who this arm belongs to?':[q11],
                      'স্কোর':[None],
                      'সম্ভাবনা-শতাংশ':[None]}
    df=pd.DataFrame(user_tracker)
    df=df.T

    df.rename(columns={
            0:f'patient{patient}'
        },inplace=True)
    score=value_mapper(df)
    df.loc['স্কোর']=score
    df.loc['সম্ভাবনা-শতাংশ']=probability_score(score)
    pscore=probability_score(score)
    
    st.write(f'আপনার স্ট্রোক হওয়ার সম্ভাবনা : {probability_score(score)}')
    finalised_data=pd.read_csv('finalised_data.csv',index_col=['Unnamed: 0'])
    col_name=df.columns
    finalised_data[col_name[0]]=df.iloc[:,0:]
    finalised_data.to_csv('finalised_data.csv')
    st.session_state.patient=st.session_state.patient + 1
    
   
    if pscore=='<15%': st.info('''
                                আমাদের বর্তমান মূল্যায়নে আপনার স্ট্রোক হবার সম্ভাবনা তুলনামূলকভাবে কম (<15%)। তবুও, আপনার স্বাস্থ্য সম্পর্কে সতর্ক থাকা গুরুত্বপূর্ণ। একটি স্বাস্থ্যকর জীবনধারা বজায় রাখুন, 
                                যেকোনো লক্ষণ নিরীক্ষণ করুন এবং আপনার ডাক্তারের সাথে নিয়মিত চেক-আপ করুন। আপনি যদি কোনো অস্বাভাবিক লক্ষণ অনুভব করেন তাহলে আপনার ডাক্তারের সাথে পরামর্শ করুন।

                            ''')
    elif score=='30%': st.info('''
                                আমাদের বর্তমান মূল্যায়নে আপনার স্ট্রোক হবার সম্ভাবনা মাঝারি (30%)। হঠাৎ শক্তি না পাওয়া, বিভ্রান্তি, দেখতে সমস্যা, হাঁটা বা তীব্র মাথাব্যথার মতো স্ট্রোকের লক্ষণ সম্পর্কে সচেতন হওয়া 
                                অত্যন্ত গুরুত্বপূর্ণ। স্বাস্থ্য পরীক্ষা/চেক আপ এবং প্রস্ট্রোক প্রতিরোধ করতে আপনার ডাক্তারের সাথে পরামর্শ করুন। যদিও এর মানে এই নয় যে আপনার স্ট্রোক হয়েছে, তবে আপনি যে লক্ষণগুলির 
                                সম্মুখীন হচ্ছেন সেজন্য আপনার চেক আপ প্রয়োজন।
                            ''')
    else: st.info('''
                    আমাদের বর্তমান মূল্যায়নে আপনার স্ট্রোক হবার সম্ভাবনা অনেক বেশি। এটি একটি মেডিকেল ইমার্জেন্সি বা জরুরি অবস্থা। আপনি যদি শরীরের একপাশে হঠাৎ শক্তি না পাওয়া, কথা বলতে অসুবিধা, মাথা ঘোরা বা 
                    ভারসাম্য হারানোর মতো কোনো লক্ষণ অনুভব করেন তবে অবিলম্বে ডাক্তারের পরামর্শ নিন। নিকটস্থ হাসপাতালে যান বা অবিলম্বে জরুরি বিভাগে যোগাযোগ করুন।
                ''')

    st.write(' ')    
    st.warning('''
                বি. দ্র :  মনে রাখবেন, এই মন্তব্যগুলো শুধুমাত্র নির্দেশনার জন্য। স্ট্রোকের লক্ষণগুলো ব্যাপকভাবে পরিবর্তিত হতে পারে এবং বিভিন্ন কা্রণে স্ট্রোকের মাত্রা উল্লেখযোগ্যভাবে প্রভাবিত হতে পারে। নিয়মিত চিকিৎসা পরামর্শ এবং 
               কোনো সতর্কতা লক্ষণের ক্ষেত্রে তাৎক্ষণিক ব্যবস্থা গ্রহণ অপরিহার্য। আমরা এই পরিমাপের জন্য আমেরিকান স্ট্রোক অ্যাসোসিয়েশন (ASA) দ্বারা একটি প্রণীত স্কেল ব্যবহার করেছি। স্ট্রোকের ঝুঁকি ব্যবস্থাপনা করার ক্ষেত্রে প্রাথমিক 
               লক্ষণ সনাক্তকরণ এবং প্রতিরোধ চাবিকাঠি।
                ''')
    st.info('ডেটা সফলভাবে সংরক্ষণ করা হয়েছে!!')
    st.markdown(get_table_download_link(finalised_data), unsafe_allow_html=True)
    

 
    

  
    
    
