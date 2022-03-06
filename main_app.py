import streamlit as st
import pandas as pd
import app


st.set_page_config(layout="wide")

# PAGES = {
#
# 	"Explore Existing Whitepapers in Database" : app,
# 	"Upload New Whitepaper" : upload
# }


def main():

	app.write()
	# selection = st.sidebar.radio("", list(PAGES.keys()))
	#
	#
	# if PAGES[selection] == app:
	# 	app.write()
	#
	# elif PAGES[selection] == upload:
	# 	upload.write()

if __name__ == '__main__':
	main()
