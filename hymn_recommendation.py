from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle


def hymn_recommender(text, hymnal_option, model, num_hymns=20):
    with open(hymnal_option + ".pkl", 'rb') as f:
        hymn_df = pickle.load(f)
        hymn_df = hymn_df.loc[hymn_df["Hymn Text"].notna()]

    text_emb = model.encode([text])
    hymnal_emb = embed_hymnal(hymn_df, nlp_model=model)

    hymn_df['Similarity'] = cosine_similarity(
        text_emb,
        hymnal_emb
    )[0]
    hymn_df['Similarity'] = hymn_df['Similarity'].apply(lambda x: np.round(x*100,1))
    return hymn_df.sort_values('Similarity', ascending=False).iloc[:,1:].head(num_hymns)


def embed_hymnal(hymnal_df, nlp_model):

    sentences = hymnal_df['Hymn Text'].values.tolist()

    sentence_embeddings = nlp_model.encode(sentences)

    return sentence_embeddings


if __name__ == '__main__':

    model_ex = SentenceTransformer('bert-base-nli-mean-tokens')
    text_ex = """
    Paul stood in front of the Areopagus and said, “Athenians, I see how extremely religious you are in every way. 
    For as I went through the city and looked carefully at the objects of your worship,
    I found among them an altar with the inscription, ‘To an unknown god.’  
    What therefore you worship as unknown, this I proclaim to you.  
    The God who made the world and everything in it, he who is Lord of heaven and earth, 
    does not live in shrines made by human hands, nor is he served by human hands, 
    as though he needed anything, since he himself gives to all mortals life and breath and all things. 
    From one ancestor he made all nations to inhabit the whole earth, 
    and he allotted the times of their existence and the boundaries of the places where they would live, 
    so that they would search for God and perhaps grope for him and find him—though indeed 
    he is not far from each one of us. For ‘In him we live and move and have our being’; 
    as even some of your own poets have said, 
    ‘For we too are his offspring.’ 
    Since we are God’s offspring, we ought not to think that the deity is like gold, 
    or silver, or stone, an image formed by the art and imagination of mortals. 
    While God has overlooked the times of human ignorance, now he commands all people everywhere to repent, 
    because he has fixed a day on which he will have the world judged in righteousness by a man whom he has appointed, 
    and of this he has given assurance to all by raising him from the dead.”
    """
    hymnal_option_ex = 'Glory to God (Presbyterian)'
    print(hymn_recommender(text_ex, hymnal_option_ex, model_ex, num_hymns=20))

