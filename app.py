import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image


from twilio.rest import Client

# Twilio credentials
ACCOUNT_SID = 'ACb826bb9045615cc2ad83014a8d27d39e'
AUTH_TOKEN = 'cea23502a12a30983e994ca9fdf3d5f9'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'  # Twilio Sandbox WhatsApp number
RECIPIENT_WHATSAPP_NUMBER = 'whatsapp:+919498089804'  # Verified recipient number
CONTENT_SID = 'HXb5b62575e6e4ff6129ad7c8efe1f983e'  # Replace with your content SID

# Function to send WhatsApp message
def send_whatsapp_message(content_variables):
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            content_sid=CONTENT_SID,
            content_variables=content_variables,
            to=RECIPIENT_WHATSAPP_NUMBER
        )
        return "success"  # Indicate success
    except Exception as e:
        return "failure"  # Indicate failure

# List of plant diseases
disease_options = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Load the model and preprocess the image
def model_predict(image_path):
    try:
        model = tf.keras.models.load_model(r"C:/Users/Lenovo/Documents/college/online courses/Edunet Foundation/CNN_plantdiseases_model.keras")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Uploaded file is not a valid image.")
        H, W, C = 224, 224, 3
        img = cv2.resize(img, (H, W))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32")
        img = img / 255.0
        img = img.reshape(1, H, W, C)

        prediction = np.argmax(model.predict(img), axis=-1)[0]
        return prediction
    except Exception as e:
        st.error(f"Error in model prediction: {e}")
        return None

# Remedies for each disease class
remedies = {
    'Apple___Apple_scab': '''Apple scab is a fungal disease that affects apple trees, causing dark, scabby lesions on leaves and fruit. 
    To manage this, apply fungicides at regular intervals, especially during the spring when the disease is most active. 
    Early treatment is crucial for preventing the spread of the disease, and removing infected leaves and fruit can help reduce the risk of reinfection.''',

    'Apple___Black_rot': '''Black rot is a serious fungal disease that causes dark, sunken lesions on the fruit, which can lead to complete rotting. 
    To manage this disease, it is essential to remove and dispose of any infected fruit and plant debris to prevent the spread. 
    Additionally, apply appropriate fungicides, especially during the flowering and fruit-setting stages. 
    Practicing crop rotation and using resistant apple varieties can further reduce the risk of infection.''',

    'Apple___Cedar_apple_rust': '''Cedar apple rust is a fungal disease that causes orange, rust-like lesions on the leaves of apple trees. 
    To manage this disease, prune affected branches to improve air circulation and remove any infected leaves from the tree. 
    It's also recommended to use rust-resistant apple varieties and apply fungicides, particularly during the spring when the spores are most active. 
    Regular monitoring and early intervention are key to preventing the spread of the disease.''',

    'Apple___healthy': '''No treatment needed for healthy plants. Keep the tree well-watered, provide adequate sunlight, and ensure proper soil nutrients for optimal growth.''',

    'Blueberry___healthy': '''No treatment needed for healthy plants. Ensure the plant is in a well-drained soil, provide proper sunlight, and monitor for pests regularly.''',

    'Cherry_(including_sour)___Powdery_mildew': '''Powdery mildew is a fungal disease that causes white, powdery spots on the leaves and stems. 
    To control this, apply fungicides and remove infected plant parts. Ensuring good air circulation and proper spacing between plants will help prevent the spread.''',

    'Cherry_(including_sour)___healthy': '''No treatment needed for healthy plants. Ensure proper care with regular watering and pest monitoring.''',

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': '''Cercospora leaf spot is a fungal disease that causes grayish lesions on corn leaves. 
    To manage this, apply fungicides and practice crop rotation to prevent reinfection. Removing infected leaves can help reduce the spread.''',

    'Corn_(maize)___Common_rust_': '''Common rust is a fungal disease that appears as reddish pustules on corn leaves. 
    To control this, apply fungicides and remove infected leaves. Regular monitoring during the growing season is essential for timely treatment.''',

    'Corn_(maize)___Northern_Leaf_Blight': '''Northern leaf blight causes long, grayish lesions on corn leaves. 
    Use resistant corn varieties and apply fungicides, particularly during the early stages of the disease. Crop rotation and proper plant spacing are also effective control measures.''',

    'Corn_(maize)___healthy': '''No treatment needed for healthy plants. Ensure the plant is in a well-drained, nutrient-rich soil, and monitor for pests and diseases regularly.''',

    'Grape___Black_rot': '''Black rot causes dark lesions on grape leaves and fruit. 
    Apply fungicides and remove infected fruit clusters. Ensure proper vine spacing and pruning to improve airflow and reduce disease pressure.''',

    'Grape___Esca_(Black_Measles)': '''Esca causes lesions on grapevines and leads to vine dieback. 
    Prune affected vines and use resistant grape varieties. Ensuring proper vine care, including avoiding over-watering and controlling pests, is also important.''',

    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': '''Leaf blight causes yellowing and necrosis of grapevine leaves. 
    Remove infected leaves and apply fungicides. Pruning for better air circulation and regular monitoring of vines are crucial for managing the disease.''',

    'Grape___healthy': '''No treatment needed for healthy plants. Regular care, such as proper pruning, adequate watering, and pest control, will keep the vine healthy.''',

    'Orange___Haunglongbing_(Citrus_greening)': '''Citrus greening is a bacterial disease that causes yellowing of leaves and premature fruit drop. 
    Use disease-free planting material and control insect vectors like aphids. Pruning infected branches and applying appropriate bactericides can help manage the disease.''',

    'Peach___Bacterial_spot': '''Bacterial spot causes lesions on peach leaves and fruit. 
    Apply copper-based fungicides and prune affected branches to improve air circulation. Avoid overhead irrigation to reduce moisture on leaves, which can promote bacterial growth.''',

    'Peach___healthy': '''No treatment needed for healthy plants. Ensure proper care with regular watering, pest monitoring, and pruning to maintain healthy growth.''',

    'Pepper,_bell___Bacterial_spot': '''Bacterial spot causes dark, water-soaked lesions on pepper leaves. 
    Apply copper-based fungicides and remove affected leaves. Ensure proper plant spacing to improve airflow and reduce disease spread.''',

    'Pepper,_bell___healthy': '''No treatment needed for healthy plants. Provide adequate sunlight, water, and nutrients for optimal growth.''',

    'Potato___Early_blight': '''Early blight causes dark lesions on potato leaves. 
    Apply fungicides and remove infected plant parts. Ensure proper plant spacing to improve airflow and reduce humidity, which promotes the spread of the disease.''',

    'Potato___Late_blight': '''Late blight is a serious fungal disease that causes rapid decay of potato plants. 
    Use resistant potato varieties and apply fungicides, especially during periods of high humidity. Removing infected plant parts can help control the spread.''',

    'Potato___healthy': '''No treatment needed for healthy plants. Ensure proper soil drainage, pest control, and adequate watering for optimal growth.''',

    'Raspberry___healthy': '''No treatment needed for healthy plants. Regular monitoring and care, such as pruning and pest control, will keep the plant healthy.''',

    'Soybean___healthy': '''No treatment needed for healthy plants. Ensure proper watering, soil nutrients, and pest control to maintain plant health.''',

    'Squash___Powdery_mildew': '''Powdery mildew causes white, powdery spots on squash leaves. 
    Apply fungicides and remove infected leaves. Improving air circulation around plants and watering at the base of the plant can help prevent the disease.''',

    'Strawberry___Leaf_scorch': '''Leaf scorch causes yellowing and browning of strawberry leaves. 
    Remove affected leaves and improve irrigation practices to ensure the plant is not stressed. Proper spacing between plants can also reduce the spread of the disease.''',

    'Strawberry___healthy': '''No treatment needed for healthy plants. Ensure proper care with adequate sunlight, water, and regular pest monitoring.''',

    'Tomato___Bacterial_spot': '''Bacterial spot causes lesions on tomato leaves and fruit. 
    Apply copper-based fungicides and remove infected leaves. Regular monitoring for pests and ensuring proper plant spacing will help manage the disease.''',

    'Tomato___Early_blight': '''Early blight causes dark, circular lesions on tomato leaves. 
    Apply fungicides and remove infected leaves. Crop rotation and proper watering practices can help reduce the spread of the disease.''',

    'Tomato___Late_blight': '''Late blight is a serious fungal disease that causes rapid plant decay. 
    Use resistant tomato varieties and apply fungicides during the growing season. Remove infected plant parts and improve airflow to reduce humidity around plants.''',

    'Tomato___Leaf_Mold': '''Leaf mold causes yellowing and mold growth on tomato leaves. 
    Remove infected leaves and improve air circulation around plants. Water at the base of the plant to avoid moisture on leaves, which promotes fungal growth.''',

    'Tomato___Septoria_leaf_spot': '''Septoria leaf spot causes dark, circular lesions on tomato leaves. 
    Apply fungicides and remove affected leaves. Crop rotation and proper watering practices can help prevent the spread of the disease.''',

    'Tomato___Spider_mites Two-spotted_spider_mite': '''Spider mites cause yellowing and speckling on tomato leaves. 
    Apply miticides and remove infested leaves. Regular monitoring and maintaining plant health will help prevent severe infestations.''',

    'Tomato___Target_Spot': '''Target spot causes dark, circular lesions with concentric rings on tomato leaves. 
    Apply fungicides and remove infected leaves. Ensure proper plant spacing to improve airflow and reduce humidity, which promotes the spread of the disease.''',

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': '''Yellow leaf curl virus causes yellowing and curling of tomato leaves. 
    Remove infected plants and control vector insects like whiteflies. Use resistant tomato varieties and practice crop rotation to reduce the risk of infection.''',

    'Tomato___Tomato_mosaic_virus': '''Tomato mosaic virus causes mottled yellow and green patterns on tomato leaves. 
    Remove infected plants and avoid plant stress. Control pests and use resistant tomato varieties to manage the disease.''',

    'Tomato___healthy': '''No treatment needed for healthy plants. Regular care, such as proper watering, pest control, and pruning, will keep the plant in optimal condition.'''
}


# Plant care tips based on plant type
plant_care_tips = {
    'Tomato': '''
    Tomatoes are sun-loving plants that require a lot of light to grow healthy. They thrive in full sunlight for at least 6 to 8 hours a day, so choose a location with maximum exposure to sunlight. The soil should be well-drained, as tomatoes do not like to sit in waterlogged soil. Consistent watering is essential, especially during dry spells, but be sure to avoid over-watering, which can lead to root rot. Tomatoes also benefit from regular feeding with a balanced fertilizer to encourage strong growth and fruit production. Additionally, pruning dead or diseased branches and supporting the plants with stakes or cages will help maintain plant health and prevent diseases.
    ''',
    'Apple': '''
    Apple trees require well-drained, slightly acidic soil to thrive. They need at least 6 hours of sunlight daily, with a preference for a sunny, open location. Regular watering is important, but it’s essential not to over-water, as apple trees are prone to root rot in soggy soil. During the growing season, apple trees should be fertilized to promote healthy growth, especially if the soil is nutrient-poor. Pruning is also important to remove dead or damaged branches and to shape the tree for better air circulation and sunlight penetration. Regular pest management, especially against apple maggots and aphids, is necessary to ensure healthy fruit production.
    ''',
    'Grape': '''
    Grapevines require a warm climate with full sunlight for optimal growth. They prefer well-drained soil with good air circulation around the plant to prevent fungal diseases like powdery mildew. Grapes thrive when their roots are well-aerated, so it’s important to plant them in loose, well-drained soil. Consistent watering is important, especially during the fruiting stage, but avoid watering the leaves to prevent fungal infections. Grapevines also benefit from regular pruning to maintain structure and promote better fruit production. For a bountiful harvest, it’s crucial to provide a trellis or support system to help the vines grow upwards and maximize sun exposure.
    ''',
    'Pepper': '''
    Peppers require warm temperatures and plenty of sunlight to grow successfully. They thrive in well-drained soil that is rich in organic matter. Peppers need consistent watering to keep the soil evenly moist, but it’s essential not to let the soil become soggy, as this can lead to root rot. For best growth, peppers should be planted in a location that receives at least 6 hours of direct sunlight daily. Peppers are sensitive to cold temperatures, so it’s important to plant them after the last frost has passed and maintain warm temperatures during their growing period. Additionally, fertilizing with a balanced nutrient mix helps promote strong plant development and abundant fruit.
    ''',
    'Potato': '''
    Potatoes require loose, well-drained soil with a slightly acidic pH to grow effectively. The soil should be rich in organic matter to provide nutrients throughout the growing season. Potatoes thrive in cooler temperatures, with an ideal range between 60°F to 70°F (15°C to 21°C). They need moderate rainfall throughout the growing season, as too much or too little water can negatively impact tuber formation. It’s important to plant potatoes in mounds or hills to improve drainage and prevent the tubers from rotting. Regular hilling, or mounding up soil around the plants, helps protect the growing tubers and encourages healthy growth. Additionally, ensure proper pest control to avoid issues with aphids and beetles.
    '''
}

# Ideal weather conditions for different plants
weather_conditions = {
    'Tomato': '''
    Tomatoes thrive in warm, moderate conditions. They grow best in temperatures between 70°F to 85°F (21°C to 29°C), which supports strong, healthy plant growth and fruit production. Tomatoes also require moderate humidity to prevent issues like fungal diseases, but excessive moisture can cause problems. They do best in regions with warm, sunny days and cool nights, which help to enhance fruit ripening. While tomatoes are quite resilient, they can be damaged by frost, so it’s important to plant them after the last frost date and provide protection if unexpected cold temperatures occur.
    ''',
    'Apple': '''
    Apples thrive in temperate climates with moderate temperatures. They grow best in temperatures ranging from 60°F to 75°F (15°C to 24°C), with cooler nights. While they need warmth during the growing season to promote healthy fruit development, apples also require a period of winter chill to produce fruit. Apple trees do well in regions with cold winters, which help to break dormancy and ensure a productive growing season. Ideal weather conditions include mild summers and winter temperatures that drop below freezing, which is essential for the tree’s cycle of growth and dormancy.
    ''',
    'Grape': '''
    Grapes are best suited for warm climates, typically thriving in temperatures between 75°F to 85°F (24°C to 29°C). These temperatures allow the grapevines to grow vigorously and produce high-quality fruit. Grapes need plenty of sunlight, as the fruit requires exposure to UV rays to ripen properly. Grapevines also need warm, dry conditions to avoid fungal diseases, which are more prevalent in damp, humid environments. While grapes need heat to ripen, they should not be exposed to extreme temperatures above 90°F (32°C), as this can stress the plant and reduce fruit quality.
    ''',
    'Pepper': '''
    Peppers thrive in warm temperatures, typically between 70°F to 85°F (21°C to 29°C). They need consistent warmth to develop healthy fruits, and the growing season should be free from frost. Peppers are sensitive to low temperatures, so planting should occur after the last frost in spring. Moderate humidity is also beneficial for peppers, but excessive moisture can lead to diseases like mildew. They prefer full sun to promote strong growth and abundant fruiting. In cooler climates, peppers may need to be started indoors and transplanted outside once the weather warms up.
    ''',
    'Potato': '''
    Potatoes prefer cooler weather and are best grown in temperatures between 60°F to 70°F (15°C to 21°C). These conditions help to produce high-quality tubers. Excessive heat can stunt tuber growth and reduce yield, so it’s important to plant potatoes early enough in the season to avoid the hottest months. Moderate rainfall is ideal for potatoes, as they require consistent moisture to form healthy tubers. However, too much rain can cause the potatoes to rot, so it’s important to ensure good drainage. Potatoes do not tolerate frost, so planting should be done after the last frost date in spring.
    '''
}


# Sidebar and page navigation
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION", "PLANT CARE TIPS", "WEATHER CONDITIONS","Send WhatsApp Notification"])

# Display image in the main page
img = Image.open(r"C:/Users/Lenovo/Documents/college/online courses/Edunet Foundation/Plant.jpg")
st.image(img, use_column_width=True)

if app_mode == "HOME":
    st.markdown(
        """
        <h1 style='text-align: center;'>Plant Disease Detection System</h1>
        <p style='text-align: center; font-size: 20px; color: #555;'>Leveraging AI to Protect Crops and Enhance Agricultural Productivity</p>
        <hr style="border:1px solid #ccc;">

        <h2 style='color: #4CAF50;'>Project Overview</h2>
        <p>This advanced Plant Disease Detection System is designed to help farmers and agricultural experts quickly identify plant diseases using artificial intelligence. By uploading an image of a plant, the system can automatically detect various diseases, providing accurate diagnoses and offering effective treatment recommendations. This technology aims to optimize crop health, reduce losses, and improve overall agricultural efficiency.</p>

        <h2 style='color: #4CAF50;'>Key Features</h2>
        <p>Our system provides fast and reliable disease identification, along with actionable insights to protect plants. It includes:</p>
        <ul>
            <li><strong>AI-powered Disease Detection:</strong> Instantly identify plant diseases from images.</li>
            <li><strong>Personalized Remedy Suggestions:</strong> Receive tailored treatment options for the identified disease.</li>
            <li><strong>Plant Care Tips:</strong> Get essential guidance for maintaining plant health and ensuring optimal growth conditions.</li>
        </ul>

        <h2 style='color: #4CAF50;'>Technology Behind the System</h2>
        <p>The core of the system is powered by cutting-edge deep learning algorithms, particularly Convolutional Neural Networks (CNNs), trained on an extensive dataset of plant diseases. These models enable precise disease classification and deliver highly accurate results, helping users take immediate and informed action to safeguard their crops.</p>

        <hr style="border:1px solid #ccc;">
        <p style='text-align: center; font-size: 16px;'>Explore the system further through the menu on the left to get started with disease detection and care recommendations.</p>
        """,
        unsafe_allow_html=True
    )





elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    save_path = None
    if test_image is not None:
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

    if st.button("Show Image") and test_image is not None:
        st.image(test_image, use_column_width=True)

    if st.button("Predict") and save_path is not None:
        st.snow()
        st.write("Processing the prediction...")
        result_index = model_predict(save_path)

        if result_index is not None:
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]
            disease = class_name[result_index]
            st.write(f"Prediction: **{disease}**")
            st.write(f"Suggested Remedy: {remedies[disease]}")
elif app_mode == "PLANT CARE TIPS":
    st.header("Plant Care Tips")
    plant_name = st.selectbox("Select a Plant:", ['Tomato', 'Apple', 'Grape', 'Pepper', 'Potato'])
    care_tip = plant_care_tips.get(plant_name, "No care tips available.")
    st.write(f"Care Tips for {plant_name}: {care_tip}")

elif app_mode == "WEATHER CONDITIONS":
    st.header("Ideal Weather Conditions for Plant Growth")
    plant_name = st.selectbox("Select a Plant:", ['Tomato', 'Apple', 'Grape', 'Pepper', 'Potato'])
    weather_tip = weather_conditions.get(plant_name, "No weather conditions available.")
    st.write(f"Ideal Weather Conditions for {plant_name}: {weather_tip}")
# Send WhatsApp message for plant disease detection
elif app_mode == "Send WhatsApp Notification":
    st.write("This section sends a WhatsApp message with the plant disease detection result.")
    
    # Display the list of diseases
    selected_disease = st.selectbox("Select Detected Disease", disease_options)
    
    # Set detection result based on selected disease
    detection_result = f"{selected_disease} detected. Follow the recommended treatment."
    
    st.write(f"Detection Result: {detection_result}")

    # Option to send the result via WhatsApp
    if st.button("Send WhatsApp Message"):
        # Create content variables dynamically
        content_variables = f'{{"1":"Plant Disease Alert","2":"{detection_result}"}}'
        result = send_whatsapp_message(content_variables)
        if result == "success":
            st.balloons()  # Display the balloon effect
            st.success("Message sent successfully!")
        else:
            st.error("Failed to send message.")






