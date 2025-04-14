import pandas as pd
from dotenv import load_dotenv
import gradio as gr
import numpy as np
from langchain_community.document_loaders import TextLoader #take raw text and convert to format langchian can work with
from langchain_text_splitters import CharacterTextSplitter #split document into meaningful chunks (individual descriptions in our case)
from langchain_openai import OpenAIEmbeddings #for embeddings
from langchain_chroma import Chroma #store in embeddings in vector databse



load_dotenv()
books = pd.read_csv("/Users/naomigong/Coding/Book_Recommender/books_with_emotion.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["thumbnail"].isna(), 
                                    "no_cover_found.jpg",
                                    books["large_thumbnail"])


#for semantic recommendations
raw_documents = TextLoader("tagged_descriptions.txt").load()
text_splitter = CharacterTextSplitter(chunk_size = 0, chunk_overlap = 0, separator = "\n")
documents = text_splitter.split_documents(raw_documents)
#creates document embeddings of each tagged description

#define way to save it
db_books = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings()
)




def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simplified_category"]==category][:final_top_k]
    else:
        book_recs.head(final_top_k)

    #sort based on probability
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending = False, inplace = True)
    elif tone == "Surprising":
        book_recs.sort_values(by="joy", ascending = False, inplace = True)
    elif tone == "Angry":
        book_recs.sort_values(by="joy", ascending = False, inplace = True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="joy", ascending = False, inplace = True)
    elif tone == "Sad":
        book_recs.sort_values(by="joy", ascending = False, inplace = True)

    return book_recs

#for gradio dashboard
def recommend_books(
        query:str,
        category: str,
        tone:str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_desc = " ".join(truncated_desc_split[:30]) + "..."
        print(row["authors"])
        if pd.isna(row["authors"]):
            print("enter")
            authors_split = "no-known-author"
        else:
            authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            #until just before last split by comma then and then and the last author
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_desc}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simplifed_categories"].unique())
tones = ["All", "Happy", "Sad", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")
    #user interaction
    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:", placeholder = "e.g., A story about forgiveness") 
        #value is the default value
        # choices = categories is the list of choices 
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    #max of 16 recommendation 8 * 2
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)
    #pass the recommend book funciton in
    #pass in the values from the user in the query and dropdowns
    #output is whatever comes out of the recommend book function
    submit_button.click(fn=recommend_books, 
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()