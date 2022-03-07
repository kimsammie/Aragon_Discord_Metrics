import streamlit as st
import pandas as pd
import json
import requests
import nltk
import re
import gensim
import spacy
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from nltk.corpus import stopwords
nltk.download('stopwords')

# st.set_page_config(layout="wide")

def write():
	st.title('Aragon Discord Channel Topics Discussed by the Community ')
	# st.subheader('by [Sammie Kim](https://www.linkedin.com/in/sammiekim/)')

	st.markdown(
	"""

	<br><br/>
	
	TBD
	
	"""
	, unsafe_allow_html=True)

	st.markdown(
	"""
	<br><br/>
	**PLEASE SELECT THE START DATE OF THE WEEK FROM THE SIDE BAR**
	"""
	, unsafe_allow_html=True)

	# df1 = pd.read_csv("plagiarison_top_selections.csv")
	# df1.sort_values('Count of score', ascending = False)
	# df2 = pd.read_csv("SBERT_raw_results.csv", index_col=False )
	# df3 = pd.read_csv("whitepaper_list.csv")
	#
	# st.sidebar.write('Choose Your Crypto Whitepapers')
	# selection = st.sidebar.selectbox('Choose Method', ['Option 1: Top # Results', 'Option 2: Manual Selection'])
	#
	# if selection == 'Option 1: Top # Results':
	#
	# 	st.sidebar.write('Option 1')
	# 	var1 = st.sidebar.number_input('Enter the number of top pairs (max 50)', max_value = 50)
	# 	st.table(df1.head(int(var1)))


	# creating two functions as discord seems to take only one request i.e., either limit or before/after message id
	# below is authorization from my discord login

	st.sidebar.write('Choose a week')
	date_ofweek = st.sidebar.date_input('Enter the the start of date the week (e.g., 2022-02-21)')
	# st.table(df1.head(int(var1)))


	st.sidebar.write('Choose the Discord channel')
	selection = st.sidebar.selectbox('Choose Method', ['Option 1: General', 'Option 2: Intro', 'Option 3: Questions'])

	if selection == 'Option 1: General':
		channel_num = '672466989767458861'
	elif selection == 'Option 2: Intro':
		channel_num = '684539869502111755'
	elif selection == 'Option 3: Questions':
		channel_num = '694844628586856469'
		# st.sidebar.write('Option 1')
		# var1 = st.sidebar.number_input('Enter the number of top pairs (max 50)', max_value = 50)
		# st.table(df1.head(int(var1)))
		#


	st.sidebar.write('Number of Topics')
	numberof_topics = st.sidebar.number_input('Enter the number of topics')
	# st.table(df1.head(int(var1)))


	def retrieve_messages1(channelid):
		headers = {
			'authorization': 'mfa.rzKIytp3I__V7txFr0cU_5VoI-pwaqlEBD0MAj3raEB5PK-imOUBFk5UnydY9Lf2eRkZAQMUfpIgPa9Mueku'
		}
		# payload={'page':2, 'count':100} # this with 'params=payload' doesn't work
		r = requests.get(f'https://discord.com/api/v9/channels/{channelid}/messages?limit=100', headers=headers)
		jsonn = json.loads(r.text)
		return jsonn

	def retrieve_messages2(channelid, messageid):
		headers = {
			'authorization': 'mfa.rzKIytp3I__V7txFr0cU_5VoI-pwaqlEBD0MAj3raEB5PK-imOUBFk5UnydY9Lf2eRkZAQMUfpIgPa9Mueku'
		}
		r = requests.get(f'https://discord.com/api/v9/channels/{channelid}/messages?before={messageid}',
						 headers=headers)
		jsonn = json.loads(r.text)
		return jsonn

		# NLTK Stop words
		# from nltk.corpus import stopwords
	stop_words = stopwords.words('english')
	stop_words.extend(
		['from', 'subject', 'use', 'not', 'would', 'say', 'could', 'be', 'know', 'good', 'go', 'get', 'do',
		 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot',
		 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come',
		 'you', 'me', 'what', 'does', 'it', 'to', 'and', 'would', 'guy', 'https', 'let', 'sure', 'set', 'maybe',
		 'still', 'able', 'look'])

	data = retrieve_messages1(channel_num)
	df = pd.DataFrame(data)
	df.sort_values('timestamp', ascending=False, inplace=True)
	df.timestamp = pd.to_datetime(df.timestamp)

	# add additional data

	while len(df) < 5000:  # or use before/after timestamp
		latestid = df.tail(1)['id'].values[0]
		newdata = retrieve_messages2(channel_num, latestid)
		df1 = pd.DataFrame(newdata)
		df1.timestamp = pd.to_datetime(df1.timestamp)
		df = pd.concat([df, df1])  # expand the database
		df.sort_values('timestamp', ascending=False, inplace=True)
	# latestdate = df.tail(1)['timestamp'].values[0]

	df = df.reset_index(drop=True)  # if not set to a variable it won't reset

	df['timestamp'] = df['timestamp'].dt.date
	start_date = pd.to_datetime('2022-02-21').date()
	end_date = pd.to_datetime('2022-02-27').date()
	one_week = (df['timestamp'] > start_date) & (df['timestamp'] <= end_date)
	df_1wk = df.loc[one_week]

	#Tokenize Sentences and Clean
	def sent_to_words(sentences):
		for sent in sentences:
			sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
			sent = re.sub('\s+', ' ', sent)  # remove newline chars
			sent = re.sub("\'", "", sent)  # remove single quotes
			sent = gensim.utils.simple_preprocess(str(sent),
												  deacc=True)  # split the sentence into a list of words. deacc=True option removes punctuations
			yield (sent)
	# Convert to list
	data = df.content.values.tolist()
	data_words = list(sent_to_words(data))


	# Build the Bigram, Trigram Models and Lemmatize
	bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
	trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	trigram_mod = gensim.models.phrases.Phraser(trigram)

	# !python3 -m spacy download en  # run in terminal once
	def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
		"""Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
		texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
		texts = [bigram_mod[doc] for doc in texts]
		texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
		texts_out = []
		nlp = spacy.load('en', disable=['parser', 'ner'])
		for sent in texts:
			doc = nlp(" ".join(sent))
			texts_out.append([token.lemma_ for token in doc if
							  token.pos_ in allowed_postags])  # to its root form, keeping only nouns, adjectives, verbs and adverbs
		# remove stopwords once more after lemmatization
		texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
		return texts_out
	data_ready = process_words(data_words)  # processed Text Data!

	#build the topic model
	# To build the LDA topic model using LdaModel(), need the corpus and the dictionary.
	# Create Dictionary
	id2word = corpora.Dictionary(data_ready)

	# Create Corpus: Term Document Frequency
	corpus = [id2word.doc2bow(text) for text in data_ready]

	# Build LDA model
	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
												id2word=id2word,
												num_topics=numberof_topics,
												random_state=100,
												update_every=1,
												chunksize=10,
												passes=10,
												alpha='symmetric',
												iterations=100,
												per_word_topics=True)

	pprint(lda_model.print_topics())  # The trained topics (keywords and weights)

	# What is the most dominant topic and its percentage contribution in each document
	# In LDA models, each document is composed of multiple topics. But, typically only one of the topics is dominant.
	# Below extracts the dominant topic for each sentence and shows the weight of the topic and the keywords.
	# It shows which document belongs predominantly to which topic.

	def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
		# Init output
		sent_topics_df = pd.DataFrame()

		# Get main topic in each document
		for i, row_list in enumerate(ldamodel[corpus]):
			row = row_list[0] if ldamodel.per_word_topics else row_list
			# print(row)
			row = sorted(row, key=lambda x: (x[1]), reverse=True)
			# Get the Dominant topic, Perc Contribution and Keywords for each document
			for j, (topic_num, prop_topic) in enumerate(row):
				if j == 0:  # => dominant topic
					wp = ldamodel.show_topic(topic_num)
					topic_keywords = ", ".join([word for word, prop in wp])
					sent_topics_df = sent_topics_df.append(
						pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
				else:
					break
		sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

		# Add original text to the end of the output
		contents = pd.Series(texts)
		sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
		return (sent_topics_df)

	df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

	# Format
	df_dominant_topic = df_topic_sents_keywords.reset_index()
	df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
	df_dominant_topic.head(10)

	# the most representative sentence for each topic

	# Get samples of sentences that most represent a given topic.
	# Display setting to show more characters in column
	pd.options.display.max_colwidth = 100

	sent_topics_sorteddf_mallet = pd.DataFrame()
	sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

	for i, grp in sent_topics_outdf_grpd:
		sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
												 grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
												axis=0)

	# Reset Index
	sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

	# Format
	sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

	# Show
	sent_topics_sorteddf_mallet.head(10)

	# a word cloud with the size of the words proportional to the weight
	# 1. Wordcloud of Top N words in each topic
	from matplotlib import pyplot as plt
	from wordcloud import WordCloud, STOPWORDS
	import matplotlib.colors as mcolors

	cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

	cloud = WordCloud(stopwords=stop_words,
					  background_color='white',
					  width=2500,
					  height=1800,
					  max_words=10,
					  colormap='tab10',
					  color_func=lambda *args, **kwargs: cols[i],
					  prefer_horizontal=1.0)

	topics = lda_model.show_topics(formatted=False)

	fig, axes = plt.subplots(numberof_topics, 2, figsize=(10, 10), sharex=True, sharey=True)

	for i, ax in enumerate(axes.flatten()):
		fig.add_subplot(ax)
		topic_words = dict(topics[i][1])
		cloud.generate_from_frequencies(topic_words, max_font_size=300)
		plt.gca().imshow(cloud)
		plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
		plt.gca().axis('off')

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.axis('off')
	plt.margins(x=0, y=0)
	plt.tight_layout()
	plt.show()


	# 	st.markdown(
	# """
	# <br><br/>
	# **IF YOU WANT A FURTHER REVIEW OF A PAIR FROM THE ABOVE TABLE, PLEASE SELECT THE PAIR FROM THE SIDE BAR**
	# """
	# , unsafe_allow_html=True)
	#
	# 	selection2 = st.sidebar.selectbox('Select a Pair for Further Review', df2['Query_Doc@Source_Doc'].unique())
	# 	st.write('Count of Score', len(df2[df2['Query_Doc@Source_Doc'] == selection2]))
	# 	st.table(df2[df2['Query_Doc@Source_Doc'] == selection2])

	# else:
	# 	st.sidebar.write('Option 2')
	# 	manual1 = st.sidebar.selectbox('Choose Whitepaper 1', df3)
	# 	manual2 = st.sidebar.selectbox('Choose Whitepaper 2', df3)
	# 	if len(df2[df2['Query_Doc@Source_Doc'] == manual1+'@'+manual2]) > 0:
	# 		st.write('**Whitepaper Pair Selected:**', manual1+'@'+manual2)
	# 		st.write('**Count of Hits Exceeding Similarity Score 80:**', len(df2[df2['Query_Doc@Source_Doc'] == manual1+'@'+manual2]))
	# 		# st.table(df2[df2['Query_Doc@Source_Doc'] == manual1+'@'+manual2].set_index('Query_Doc@Source_Doc'))
	# 		st.table(df2[df2['Query_Doc@Source_Doc'] == manual1+'@'+manual2].sort_values(by='Cosine_Similarity_Score', ascending=False))
	# 	else:
	# 		st.write('**Whitepaper Pair Selected:**', manual2+'@'+manual1)
	# 		st.write('**Count of Hits Exceeding Similarity Score 80:**', len(df2[df2['Query_Doc@Source_Doc'] == manual2+'@'+manual1]))
	# 		st.table(df2[df2['Query_Doc@Source_Doc'] == manual2+'@'+manual1].sort_values(by='Cosine_Similarity_Score', ascending=False))
	#


	st.sidebar.write('[Source Code](https://github.com/kimsammie/plagiarism)')