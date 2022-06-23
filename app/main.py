# Run by typing python3 main.py

# **IMPORTANT:** only collaborators on the project where you run
# this can access this web server!

"""
    Bonus points if you want to have internship at AI Camp
    1. How can we save what user built? And if we can save them, like allow them to publish, can we load the saved results back on the home page? 
    2. Can you add a button for each generated item at the frontend to just allow that item to be added to the story that the user is building? 
    3. What other features you'd like to develop to help AI write better with a user? 
    4. How to speed up the model run? Quantize the model? Using a GPU to run the model? 
"""

# import basics
import os

# import stuff for our web server
from flask import Flask, request, redirect, url_for, render_template, session
from utils import get_base_url
# import stuff for our models
from aitextgen import aitextgen
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import snscrape.modules.twitter as sntwitter
import pandas as pd
import json
import plotly
import plotly.express as px
# load up a model from memory. Note you may not need all of these options.
ai = aitextgen(model_folder="model/trained_model", to_gpu=False)

# ai = aitextgen(model="distilgpt2", to_gpu=False)
stop_words = stopwords.words('english')
stop_words.remove('over')
stop_words.remove('above')
stop_words.remove('below')
stop_words.remove('up')
stop_words.remove('down')
stop_words.remove('under')
stop_words.append('https')


def remove_stopwords(sentences):
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [word for word in words if word.lower() not in stop_words]
        sentences[i] = ' '.join(words)
    return sentences

# print('running stopwords')
def remove_punctuation(sentences):
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [word for word in words if word.isalnum()]
        sentences[i] = ' '.join(words)
    return sentences

# print('running punct')
def remove_nums(sentences):
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        new_result = []
        for word in words:
            result = ''.join([letter for letter in word if not letter.isdigit()])
            if result != ' ':
                new_result.append(result)
        sentences[i] = ' '.join(new_result)
    return sentences

# print('running nums')
def remove_ticker(sentences):
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [word for word in words if not word == word.upper()]
        sentences[i] = ' '.join(words)
    return sentences

# print('running ticker')
#print(userPrompt)





# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)


# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

app.secret_key = os.urandom(64)

# set up the routes and logic for the webserver


@app.route(f'{base_url}')
def home():
    return render_template('writer_home.html', generated=None)


@app.route(f'{base_url}', methods=['POST'])
def home_post():
    return redirect(url_for('results'))


@app.route(f'{base_url}/revamped_results/')
def revamped_results():
    if 'revamped_data' in session:
        revamped_data = session['revamped_data']
        return render_template('new_mod.html', generated=revamped_data)
    else:
        return render_template('new_mod.html', generated='hi')



@app.route(f'{base_url}/new_model/', methods=['POST'])
def new_model():
    ticker = request.form['prompt']
    if ticker is not None:
        query = f"{ticker} lang:en"
        tweets = []
        limit = 100
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            if len(tweets) == limit:
                break
            else:
                tweets.append([tweet.date, tweet.content])
        df = pd.DataFrame(tweets, columns=['Date', 'Content'])
        df['Content'] = df['Content'].replace('\n',' ', regex=True)
        df['Content'] = df['Content'].replace('#', ' ')
        df['Content'] = df['Content'].replace('$', ' ')
        new_list =[]
        new_date_list = []
        count = 0
        
        for thing, date in zip(df["Content"], df["Date"]):
            count = 0
            things = thing.split(' ')
            for i in range(len(things)):
                if things[i] == things[i].upper() and len(things[i]) > 1:
                    if things[i] != ticker and things[i] not in ['EST', 'PST', 'CST', 'MST', 'GMT', 'STOCK', 'NASDAQ', 'SPX']:
                        count += 1
                    else:
                        pass
            if count == 0:
                new_list.append(thing)
                new_date_list.append(date)
        
        new_list = remove_stopwords(new_list)
        new_list = remove_punctuation(new_list)
        new_list = remove_nums(new_list)
        new_list = remove_ticker(new_list)
        sentiments = []
        for twt in new_list:
            generated = ai.generate(
                    n=1,
                    batch_size=3,
                    prompt=str(twt+';'),
                    max_length=40,
                    temperature=0.9,
                    return_as_list=True
                )
            generated = (generated[0].split(';')[1])
            generated = generated.split('\n')[0]
            sentiments.append(generated)
        #twt_dict= {'Date': df['Date'], 'Content': new_list, 'Sentiments': sentiments}
        twt_dict= {'Date': new_date_list, 'Content': new_list, 'Sentiments': sentiments}
        new_df = pd.DataFrame(twt_dict)
        data = new_df
        data.Date = pd.to_datetime(data.Date)
#         data2 = (data.set_index('Date')
#                     .groupby('Sentiments')
#                     .resample('H')
#                     .size())
        temp_df = pd.get_dummies(data, columns = ['Sentiments']).set_index('Date').resample('15M').sum().reset_index()
        graph = px.line(temp_df, x = 'Date', y = 'Sentiments_positive', title="Graph")
        graph.add_scatter(x=temp_df['Date'], y=temp_df['Sentiments_negative'], mode='lines', name='Negative')
        graph.update_layout(
            title="Number of Positive and Negative Tweets",
            xaxis_title="Time",
            yaxis_title="Number of Tweets",
            legend_title="",
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="Black"
            )
        )
#         data2 = data2.reset_index()s
        
#         graph = px.line(data2, x = data2.index[0], y = ['positive'], title="Graph")
        #graph.add_scatter(x=data2.index, y=data2['negative'], mode='lines')
        graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
        data_2 = {'generated_lsa': graphJSON}
        session['revamped_data'] =  graphJSON
        return(redirect(url_for('revamped_results')))


@app.route(f'{base_url}/results/')
def results():
    if 'data' in session:
        data = session['data']
        return render_template('Write-your-story-with-AI.html', generated=data)
    else:
        return render_template('Write-your-story-with-AI.html', generated=None)


@app.route(f'{base_url}/generate_text/', methods=["POST"])
def generate_text():
    """
    view function that will return json response for generated text. 
    """

    prompt = request.form['prompt']
    if prompt is not None:
        prompt = sent_tokenize(prompt)
        prompt = remove_stopwords(prompt)
        prompt = remove_punctuation(prompt)
        prompt = remove_nums(prompt)
        prompt = remove_ticker(prompt)
        for wrd in prompt:
            if wrd == '':
                prompt.remove(wrd)
        prompt = ' '.join(prompt)
        prompt += ';'
        generated = ai.generate(
            n=1,
            batch_size=3,
            prompt=str(prompt),
            max_length=300,
            temperature=0.9,
            return_as_list=True
        )
    generated = (generated[0].split(';')[1])
    generated = generated.split('\n')[0]
    data = {'generated_ls': generated}
    session['data'] = "The sentiment for this tweet is " + str(generated)
    return redirect(url_for('results'))

# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

############ EXTRA PREPROCESSING FUNCTIONS








if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalcg9.ai-camp.dev'

    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host='0.0.0.0', port=port, debug=True)
