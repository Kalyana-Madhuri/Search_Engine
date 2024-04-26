import re
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="/search_engine_db")
collection = client.get_collection(name="search_engine")
collection_name = client.get_collection(name="search_engine_FileName")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to clean data
def clean_data(data):
    # Remove timestamps
    data = re.sub("\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}", " ",  data)
    # Remove index no. of dialogues
    data = re.sub(r'\n?\d+\r', "", data)
    # Remove escape sequences like \n \r
    data = re.sub('\r|\n', "", data)
    # Remove <i> and </i>
    data = re.sub('<i>|</i>', "", data)
    # Remove links
    data = re.sub("(?:www\.)osdb\.link\/[\w\d]+|www\.OpenSubtitles\.org|osdb\.link\/ext|api\.OpenSubtitles\.org|OpenSubtitles\.com", " ", data)
    # Convert to lower case
    data = data.lower()
    return data

# Function to extract IDs
def extract_id(id_list):
    new_id_list = []
    for item in id_list:
        match = re.match(r'^(\d+)', item)
        if match:
            extracted_number = match.group(1)
            new_id_list.append(extracted_number)
    return new_id_list

# Streamlit UI
st.header("Movie Subtitle Search Engine")
search_query = st.text_input("Enter a dialogue to search....")
if st.button("Search"):
    st.subheader("Relevant Subtitle Files")
    search_query = clean_data(search_query)
    query_embed = model.encode(search_query).tolist()
    search_results = collection.query(query_embeddings=query_embed, n_results=10)
    id_list = search_results['ids'][0]
    id_list = extract_id(id_list)
    for id in id_list:
        file_name = collection_name.get(ids=f"{id}")["documents"][0]
        st.markdown(f"[{file_name}](https://www.opensubtitles.org/en/subtitles/{id})")