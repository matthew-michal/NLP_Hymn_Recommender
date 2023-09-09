import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
from hymn_recommendation import hymn_recommender

model = SentenceTransformer('bert-base-nli-mean-tokens')

# Top of page
st.title('Hymn Recommendation')
st.write("Enter some text, select your desired hymnal, and click on 'Calculate' for this model to recommend hymns based on your text.")

# Text area
text = st.text_area("Enter text here!", height=300)


# Hymnal choice
hymnal_option = st.selectbox(
    "Which hymnal would you wish to compare your text to?",
    ('Glory to God (Presbyterian)', "The Hymnal 1982 (Episcopal)",
     "Lift Every Voice and Sing II (Episcopal)", "The United Methodist Hymnal (1989)")
)
st.write("You selected:", hymnal_option)

# Show image of hymnal
image = Image.open(hymnal_option + (".jpg" if hymnal_option != "Lift Every Voice and Sing II (Episcopal)" else ".gif"))
st.image(image, caption=hymnal_option)
st.write("This image of your chosen hymnal comes from Hymnary.org.")

# Button to calculate word embedding similarities
if st.button("Calculate"):
    if text:
        st.dataframe(hymn_recommender(text, hymnal_option, model, num_hymns=20))
    else:
        st.write("Oops! You forgot to tell what to analyze. Please write/copy your text in the above text box.")




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
