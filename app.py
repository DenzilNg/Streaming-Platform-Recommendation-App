import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
import openai
from PIL import Image
import requests
import base64
import pandas as pd
import pickle
import sklearn

# Set the page configuration
st.set_page_config(
    page_title="Streaming Platform Recommendation",
    page_icon="🎬",  # Can be an emoji or path to an image file
    layout="wide",   # "centered" (default) or "wide"
    initial_sidebar_state="expanded",  # "collapsed" or "expanded"
)

st.title("Welcome to the Streaming Platform Recommendation App")

# Display a .webp image in Streamlit
st.image('images/chatbot.webp', width = 300)

# Create tabs
tab1, tab2= st.tabs(["Streaming Platform Recommender", "Streaming Platform CHATBOT"])

#content for tab1
with tab1:
    # Load the dataset (for column reference)
    file_path = 'data/user.csv'
    data = pd.read_csv(file_path)

    # Set input widgets
    st.sidebar.subheader('Input values below')
    Age = st.sidebar.slider('What is your age?', 0, 80, 30)
    Price = st.sidebar.number_input('How much are you willing to pay for the streaming platform?', min_value=0, max_value=25, value=10)
    Average_IMDB_ratings = st.sidebar.number_input('On a scale of 0-10 rate your preference for high IMDb ratings', min_value=0, max_value=10, value=5)
    Original_shows_and_movies = st.sidebar.number_input('On a scale of 0-10 rate your preference for Original shows and movies', min_value=0, max_value=10, value=5)
    Asian_movies_and_shows = st.sidebar.number_input('On a scale of 0-10 rate your preference for Asian shows and movies', min_value=0, max_value=10, value=5)
    Japanese_anime = st.sidebar.number_input('On a scale of 0-10 rate your preference for Japanese Anime', min_value=0, max_value=10, value=5)
    US_animation = st.sidebar.number_input('On a scale of 0-10 rate your preference for US animation', min_value=0, max_value=10, value=5)
    Superheroes_shows_and_movies = st.sidebar.number_input('On a scale of 0-10 rate your preference for Superheroes shows and movies', min_value=0, max_value=10, value=5)
    Documentaries = st.sidebar.number_input('On a scale of 0-10 rate your preference for documentaries', min_value=0, max_value=10, value=5)
    Platform_popularity = st.sidebar.number_input('On a scale of 0-10 rate your likelyhood to be influenced by platform popularity', min_value=0, max_value=10, value=5)
    Total_quantity_of_shows_and_movies = st.sidebar.number_input('On a scale of 0-10 rate your likelyhood to be influenced by total quanity of shows and movies on the platform', min_value=0, max_value=10, value=5)
    Old_movies_and_shows_before_year_2000 = st.sidebar.number_input('On a scale of 0-10 rate your preference for old movies and shows before the year 2000', min_value=0, max_value=10, value=5)

    # Load the trained model from the pickle file
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Convert input data into the correct format (e.g., a list or numpy array)
    input_data = [[Age, Price, Average_IMDB_ratings, Original_shows_and_movies, Asian_movies_and_shows, Japanese_anime, US_animation, Superheroes_shows_and_movies, Documentaries, Platform_popularity, Total_quantity_of_shows_and_movies, Old_movies_and_shows_before_year_2000]]

    # Define a mapping from numeric classes to meaningful names
    class_mapping = {0: 'Amazon Prime Video', 1: 'Disney+', 2: 'Netflix'}

    # Button to trigger prediction
    if st.sidebar.button('Recommendation'):
        prediction = model.predict(input_data)  # Use the loaded model to make a prediction
        recommended_platform = class_mapping[prediction[0]]  # Map the numeric prediction to the platform name
        st.write(f'## Your Recommended Streaming Platform is: {recommended_platform}')

        # Display the image based on the prediction
        if recommended_platform == 'Netflix':
            image_path = 'images/netflix.png'
        elif recommended_platform == 'Amazon Prime Video':
            image_path = 'images/amazon_prime_video.png'
        elif recommended_platform == 'Disney+':
            image_path = 'images/disney+.png'
        
        # Center the image using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            pass
        with col2:
            st.image(image_path, width=500)  # Adjust the width to 500
        with col3:
            pass
    

#content for tab2
with tab2:
    #Context

    # Set OpenAI API key
    openai.api_key = st.secrets.openai_key


    #Prepare a line while users are waiting for their results
    @st.cache_resource(show_spinner=False)
    def load_data():
        with st.spinner(text="Loading your recommedation... Think about your favourite shows! This may take 1-2 minutes."):
            
            # Rebuild the storage context
            storage_context = StorageContext.from_defaults(persist_dir="./data/index.vecstore")
    
            # Load the index
            index = load_index_from_storage(storage_context)
    
            # Load the finetuned model 
            ft_model_name = "gpt-4"
            ft_context = ServiceContext.from_defaults(llm=OpenAI(model=ft_model_name, temperature=0.3), 
            context_window=2048, 
            
            system_prompt="""
           Craft a series of insightful and relevant questions that potential customers might ask about streaming service platforms like Netflix, Disney+ and Amazon Prime Video to help them decide which streaming service platforms to use. You have multiple preferences regarding the type of shows and movies and want to ensure that the streaming service platform can match those preferences. You want to choose a streaming service platform that fits your age and price budget as well as preference for IMDB ratings, original shows, asian shows, japanese anime, animation, superheroes, documentaries and old movies. You would also want to choose a streaming service platform that fit your preference for streaming service platform's popularity, total number of shows and movies on the platform. You are particularly interested in which streaming service platform to use based on these preferences. Your goal is to recommend the most suitable streaming service platform to users based on their preferences. You are looking for a definite recommendation of streaming service platform that aligns with user preferences.
           Give the correct recommendation.
            """
            )           
            return index
    
    index = load_data()
    chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)
    
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask Me Questions Relating to Streaming Platforms 😊"}
        ]
    
    if prompt := st.chat_input("Ask Me Questions Relating to Streaming Platforms"):
        # Save the original user question to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.new_question = True
    
        # Create a detailed prompt for the chat engine
        chat_history = ' '.join([message["content"] for message in st.session_state.messages])
        detailed_prompt = f"{chat_history} {prompt}"
    
    if "new_question" in st.session_state.keys() and st.session_state.new_question:
       for message in st.session_state.messages: # Display the prior chat messages
           with st.chat_message(message["role"]):
               st.write(message["content"])
       st.session_state.new_question = False # Reset new_question to False
    
    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
       with st.chat_message("assistant"):
           with st.spinner("Calculating..."):
               response = chat_engine.chat(detailed_prompt)
               st.write(response.response)
               # Append the assistant's detailed response to the chat history
               st.session_state.messages.append({"role": "assistant", "content": response.response})
