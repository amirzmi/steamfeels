import streamlit as st
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.ensemble import RandomForestClassifier
from lime import lime_text
import re

import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px

from streamlit_metrics import metric, metric_row

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

st.set_page_config(page_icon="♨️", 
                   page_title="SteamFeels - Sentiment Analyser For Reviews On Steam Platform", 
                   layout="wide", 
                   initial_sidebar_state="expanded",
                   #menu_items={'Get Help': 'https://github.com/santanukumar666/Twitter-Sentiment-Analysis', 
                               # 'About': "## A Twitter sentiment analysis webappTwitter Sentment Analysis Web App using #Hashtag and username to fetch tweets and tells the sentiment of the perticular #Hashtag or username."} 
                )

fig = go.Figure()

@st.cache_resource

# Sentiment Analyser Functions
def load_vect_and_model(): # called trained model
    text_vectorizer = load("Model/vec.joblib")
    classif = load("Model/clf.joblib")
    
    return text_vectorizer, classif

text_vectorizer, classif = load_vect_and_model()

def vectorize_text(texts): # vectorizer
    text_transformed = text_vectorizer.transform(texts)
    return text_transformed

def pred_class(texts): # predict sentiment using model
    sentiment = []
    sentiment = classif.predict(vectorize_text(texts))
    return sentiment

def pred_probs(texts):
    return classif.predict_proba(vectorize_text(texts))

def create_colored_review(review, word_contributions):
    tokens = re.findall(text_vectorizer.token_pattern, review)
    modified_review = ""
    for token in tokens:
        if token in word_contributions["Word"].values:
            idx = word_contributions["Word"].values.tolist().index(token)
            conribution = word_contributions.iloc[idx]["Contribution"]
            modified_review += ":green[{}]".format(token) if conribution>0 else "{}".format(token) if conribution==0 else ":red[{}]".format(token)
            modified_review += " "
        else:
            modified_review += token
            modified_review += " "
            
    return modified_review

explainer = lime_text.LimeTextExplainer(class_names=classif.classes_)

#t1, t2 = st.columns([4,7])
#with t1: 
#    st.title(":blue[SteamFeels] |")
#with t2:
#   st.markdown("### Sentiment Analyser For Reviews On Steam Platform")

# Web App/ Dashboard code
st.markdown("<h1 style='text-align: center;'>  🆂🆃🅴🅰🅼🅵🅴🅴🅻🆂 </h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'> 🙂 Sentiment Analyser For Reviews On Steam Platform 🙁</h5>", unsafe_allow_html=True)
st.markdown("")
#st.divider()
# st.caption("App created by [Amir Azmi](https://www.linkedin.com/in/amir-azmi-064a62261/). Find the source code for this project in the [Amirzmi GitHub Repository](https://github.com/amirzmi/).")

tab1, tab2= st.tabs(["Single Reviews Analysis", "Multiple Reviews Analysis"])

uploaded_file = None

 # single anlysis Interface
with tab1:
    review = st.text_area(label="Enter Your Review Here: ", value="This game is fun", height=20)
    submit = st.button("Predict")

    # with st.spinner('Processing your review ... '):
   
    if submit and review:
        col1, col2 = st.columns([2,3])

        prediction, probs = pred_class([review]), pred_probs([review])
        prediction, probs = prediction[0], probs[0]

        with col1:
            if prediction == 1:
                prediction = "Positive"
                st.markdown("### Prediction : :green[{}]".format(prediction))
            else:
                prediction = "Negative"
                st.markdown("### Prediction : :red[{}]".format(prediction))

            st.metric(label="Confidence", value ="{:.2f}%".format(probs[1]*100 if prediction=="Positive" else probs[0]*100),
                    delta="{:.2f}%".format((probs[1]*100-50) if prediction=="Positive" else (probs[0]*100-50))
                    )
            
            explanation = explainer.explain_instance(review, classifier_fn=pred_probs, num_features=50)
            word_contribution = pd.DataFrame(explanation.as_list(), columns=["Word", "Contribution"])
            modified_review = create_colored_review(review, word_contribution)
            st.write(modified_review)

        with col2:
            fig = explanation.as_pyplot_figure()
            fig.set_figheight(7)
            st.pyplot(fig, use_container_width=True)

with tab2:
    st.subheader("📑 Upload your Reviews files")
    uploaded_file = st.file_uploader(label="upload here", type=("txt","csv"),label_visibility="collapsed")

    if uploaded_file is not None:

        #with st.spinner('Getting data from the files...'):
        #    st.success('🎈Filename: {}'.format(uploaded_file.name)+', has been load successfully. ')

        count_positive = 0
        count_negative = 0
        count_neutral  = 0

        input_df = pd.read_csv(uploaded_file)

        st.subheader(' 📜 Summary of :green[{}]'.format(uploaded_file.name))

        pred=[]
        conf=[]
        for i in range(input_df.shape[0]):
            
            review = str(input_df['review'].iloc[i])

            prediction, probs = pred_class([review]), pred_probs([review])
            prediction, probs = prediction[0], probs[0]
            
            if prediction == 1:
                prediction = "Recommended"
                count_positive+=1
                count= i+1
                #st.markdown("### Prediction : :green[{}]".format(prediction))
                # input_df['Sentiment'] = pd.concat([input_df['Sentiment'], pd.DataFrame([prediction])], ignore_index=True)
                pred.append(prediction)
                conf.append("{:.2f}%".format(probs[1]*100 if prediction=="Recommended" else probs[0]*100))
            else:
                prediction = "Not Recommended"
                count_negative+=1
                count= i+1
                #st.markdown("### Prediction : :red[{}]".format(prediction))
                #input_df = pd.concat([input_df, pd.DataFrame([prediction])], ignore_index=True)
                pred.append(prediction)
                conf.append("{:.2f}%".format(probs[1]*100 if prediction=="Recommended" else probs[0]*100))
            
        input_df['Sentiment'] = pd.Series(pred)
        input_df['Confidence'] = pd.Series(conf)

        # Summary reviews
        metric_row(
            {
                "Total Reviews": "{}".format(count),
                "% 😃 Positive Reviews": "{}".format(count_positive) + " | {:.0%}".format(count_positive/count),
                "% 😡 Negatve Reviews" : "{}".format(count_negative) + " | {:.0%}".format(count_negative/count)   
            }
        ) 

        # Set Distribution of All Sentiment interface
        title = " 🎭 Sentiment Distribution of :green[{}]".format(uploaded_file.name)
        st.subheader(title)             
        x = ["Recommended", "Not Recommended"]
        y = [count_positive, count_negative]
        row55_spacer1, row55_1, row55_spacer2 = st.columns((.2, 10, .2))
        
        with row55_1: 
            import plotly.graph_objects as go

            labels = ['Recommended','Not Recommended']
            values = [count_positive, count_negative]
            colors = ['#1F77B4','#FF7F0E']

            fig2 = go.Figure(data=[go.Pie(labels=labels, values=values)])
            fig2.update_traces(textposition='inside',hoverinfo='label+value+percent', textinfo='label+percent', textfont_size=14,
                  marker=dict(colors=colors, line=dict(color='aliceblue', width=2)))
            st.plotly_chart(fig2, use_container_width=True)

        # Set Wordcloud interface
        st.subheader('☁️ Wordcloud of :green[{}]'.format(uploaded_file.name))
        wordcloud_expander = st.expander('Expand to customize wordcloud', expanded=False)
        wordcloud_expander.subheader('Advanced Settings')
        wordcloud_words = 15

        with wordcloud_expander.form('form_2'):    
            score_type= st.selectbox('Select sentiment', ['All', 'Recommended', 'Not Recommended'], key=1)
            wordcloud_words = st.number_input('Choose the max number of words for the word cloud', 15, key = 3)
            submitted2 = st.form_submit_button('Regenerate Wordcloud', help = 'Re-run the Wordcloud with the current inputs')

        if submitted2 is not True:
            score_type = 'All'
            out = input_df.review
        # Scenario 2: All
        if score_type == 'All':
            out = input_df.review
        # Scenario 3: Positive
        if score_type == 'Recommended':
            out = input_df[input_df['Sentiment']=="Recommended"].review
        # Scenario 5: Negative
        if score_type == 'Not Recommended':

            out = input_df[input_df['Sentiment']=="Not Recommended"].review

        #Word Cloud
        st.set_option('deprecation.showPyplotGlobalUse', False)
        words = " ".join(rev for rev in out)
        #words = " ".join(word for review in input_df for word in input_df.split())
        st_en = set(stopwords.words('english'))
        wordcloud = WordCloud(stopwords=st_en, background_color="white", max_words = wordcloud_words, width=600, height=300)
        wordcloud.generate(words)
        fig2, ax2 = plt.subplots(figsize=(5,5))
        ax2.axis("off")
        fig2.tight_layout(pad=0)
        ax2.imshow(wordcloud, interpolation='bilinear')
        st.pyplot(fig2)

        def get_top_n_gram(tweet_df, ngram_range, n=10):

            st_en = set(stopwords.words('english'))

            # load the corpus and vectorizer
            corpus = tweet_df
            vectorizer = text_vectorizer(
                analyzer="word", ngram_range=ngram_range, stop_words=st_en
            )

            # use the vectorizer to count the n-grams frequencies
            X = vectorizer.fit_transform(corpus.astype(str).values)
            words = vectorizer.get_feature_names_out()
            words_count = np.ravel(X.sum(axis=0))

            # store the results in a dataframe
            df = pd.DataFrame(zip(words, words_count))
            df.columns = ["words", "counts"]
            df = df.sort_values(by="counts", ascending=False).head(n)
            df["words"] = df["words"].str.title()
            return df
        
        def plot_n_gram(n_gram_df, title, color="#54A24B"):
            # plot the top n-grams frequencies in a bar chart
            fig = px.bar(
                x=n_gram_df.counts,
                y=n_gram_df.words,
                title="<b>{}</b>".format(title),
                text_auto=True,
            )
            fig.update_layout(plot_bgcolor="white")
            fig.update_xaxes(title=None)
            fig.update_yaxes(autorange="reversed", title=None)
            fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color)
            return fig
        
        # plot the top 10 occuring words 
        #top_unigram = get_top_n_gram(out, ngram_range=(1, 1), n=10)
        #unigram_plot = plot_n_gram(
        #    top_unigram, title="Top 10 Occuring Words", color="green"
        #)
       # unigram_plot.update_layout(height=350)
        #st.plotly_chart(unigram_plot, theme=None, use_container_width=True)


        # plot the top 10 occuring bigrams
        #top_bigram = get_top_n_gram(out, ngram_range=(2, 2), n=10)
        #bigram_plot = plot_n_gram(
        #    top_bigram, title="Top 10 Occuring Bigrams", color="green"
        #)
        #bigram_plot.update_layout(height=350)
        #st.plotly_chart(bigram_plot, theme=None, use_container_width=True)

        # Set Interpretation interface
        st.subheader(' 🗣 Interpretation on :green[{}]'.format(uploaded_file.name))

        if count_positive>count_negative:
            st.markdown(" ✅ Great work there! roughly {} ".format(count_positive) + 
                        " out of every {} users ".format(count)+
                        " (or roughly {:.0%}) ".format(count_positive/count)+
                        " gave positive reviews! " + 
                        " \n ✅ That means the majority of people recommended the games. 😃")
        elif count_negative>count_positive:
            st.markdown(" ❌ Try improving your games! roughly {} ".format(count_negative) + 
                        " out of every {} users ".format(count)+
                        " (or roughly {:.0%}) ".format(count_negative/count)+
                        " gave negative reviews! " + 
                        " \n ❌ The majority of users didn't recommend your games up to the mark. 😶")
        else:
            st.markdown("""🆗 Good Work there, but there's room for improvement! Majority of people have neutral reactions. 😶""") 

        see_data = st.expander('## Expand to see the reviews sentiment prediction and its confidence. ')
        # Set review sentiment prediction and confidence interface
        with see_data:
            #st.dataframe(input_df['review']) 
            gb = GridOptionsBuilder.from_dataframe(input_df)
            gb.configure_side_bar()
            gb.configure_pagination()
            grid_options = gb.build()
            AgGrid(
                input_df,
                gridOptions=grid_options,
                enable_enterprise_modules=True,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                fit_columns_on_grid_load=True,
            )

st.sidebar.title("🆂🆃🅴🅰🅼🅵🅴🅴🅻🆂")
st.sidebar.markdown("App created by [Amir Azmi](https://www.linkedin.com/in/amir-azmi-064a62261/).")
st.sidebar.markdown("")
st.sidebar.header("❓About App")
st.sidebar.markdown('☑ Sentiment Analyser For Reviews On Steam Platform. \n\n ☑ The main purpose of this app is to provide valuable insights of sentiments expressed in Steam reviews.')

st.sidebar.markdown("")
st.sidebar.header("📝 Rate this Application ")
feedback = st.sidebar.slider('How much would you rate this app?',min_value=0,max_value=10,step=1)
if feedback:
  st.sidebar.markdown("Thank you for rating the app!")