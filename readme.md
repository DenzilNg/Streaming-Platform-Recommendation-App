# Streaming Platform Recommendation App
This project aims to provide users with a smart recommendation system for selecting the most suitable streaming platform based on their preferences. Instead of subscribing to multiple platforms, users can use this app to identify the one that best matches their content interests and budget, potentially saving time and money.

## Problem Statement
With the growing number of streaming platforms and the rising costs of subscription, many users find it challenging to justify paying for multiple services, especially given limited time to consume content. A recent Deloitte study highlighted that American households pay an average of $61 per month for four services, many of which they don’t feel are worth the cost.

## The Problem:
Multiple streaming platforms, increasing subscription costs, and limited viewing time.

## The Solution:
Streaming Platform Recommendation App: An app that recommends the best streaming platform based on a user’s preferences (e.g., age, price sensitivity, favorite genres like anime, US animation, Asian shows, etc.).

### Features
#### 1. Classification Modelling
A classification model built using user preferences such as:

Age
Price sensitivity
Average IMDb ratings
Original shows and movies
Genre preferences (Anime, Asian shows, US Animation, Superheroes shows, Documentaries, Old movies and shows before the year 2000)
Platform popularity
Total quantity of shows and movies	


The model recommends one of three major platforms:

Netflix
Disney+
Amazon Prime Video

3 classification models were used. The models used as well as the results gotten can be view here:
[View Streaming platform classification model]([https://github.com/D3nz1ll/Denzil_capstone_chatbot_GitHub/blob/main/2._Streaming_platform_classification_model.ipynb](https://github.com/D3nz1ll/Denzil_capstone_chatbot_GitHub/blob/main/2.%20Streaming%20platform%20classification%20model.ipynb))

#### 2. AI Chatbot Integration
An AI chatbot using gpt models has been trained using a questions & answers dataset, allowing users to interact and receive answers about platform features, pricing, and recommendations.

3 gpt models were used. The models used as well as the results gotten can be view here:
[3. Streaming CHATBOT model](https://github.com/D3nz1ll/Denzil_capstone_chatbot_GitHub/blob/main/3._reaming_CHATBOT.ipynb)

## Demo
The app demonstrates how users can input their preferences and see which streaming platform is most suitable for them. It also integrates a chatbot for interactive queries.The demo app can be viewed here:
[View Streamlit App](https://denzilcapstone-recommenderandchatbot.streamlit.app/)


## Future Work
1. Add more streaming platforms for consideration beyond Netflix, Disney+, and Amazon Prime.
2. Expand the AI chatbot dataset to include more detailed and relevant questions users may ask.
3. Include a Movie/Show Recommendation feature that suggests top content on the selected platform based on user preferences.


## Impact
By providing tailored recommendations, the app aims to save users money and time by reducing the need for multiple streaming subscriptions. This app helps users make more informed decisions about their streaming service subscriptions, reducing unnecessary expenses and helping them find content they enjoy on a single platform.
