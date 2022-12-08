
import streamlit as st
import numpy as np
import pandas as pd
import json
import base64
import uuid
from pandas import DataFrame
import time
import re

def download_button(object_to_download, download_filename, button_text):

    if isinstance(object_to_download, bytes):
        pass

    elif isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f"""
    <style>
    #{button_id} {{
                     display: inline-flex;
                     align-items: center;
                     justify-content: center;
                     background-color: rgb(255, 255, 255);
                     color: rgb(38, 39, 48);
                     padding: .25rem .75rem;
                     position: relative;
                     text-decoration: none;
                     border-radius: 4px;
                     border-width: 1px;
                     border-style: solid;
                     border-color: rgb(230, 234, 241);
                     border-image: initial;
                     }}
                     #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
                }}
                #{button_id}:active {{
                                 box-shadow: none;
                                 background-color: rgb(246, 51, 102);
                                 color: white;
                                 }}
                                 </style> """

    dl_link = (
        custom_css
        + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
    )
    # dl_link = f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}"><input type="button" kind="primary" value="{button_text}"></a><br></br>'

    st.markdown(dl_link, unsafe_allow_html=True)



class c_model:
    def __init__(self):
        # st.write('my model')
        pass

    @st.cache
    def load_model(self, name_or_path):
        time.sleep(3)
        return None

    def predict(self, texts):
        return np.random.randint(2), np.random.rand()


st.title('Comment Analysis')



allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'
punct = '!?,.@#'
maxlen = 80

def preprocess(text):
    return ''.join([' ' + char + ' ' if char in punct else char for char in [char for char in re.sub(r'http\S+', 'http', text, flags=re.MULTILINE) if char in allowed_chars]])[:maxlen]


selected_model = st.radio(
    'Choose model',
    ['XLM-RoBERTa', 'ParsBERT', 'ALBERT'],
    horizontal=True)


# Load classification model
with st.spinner('Loading classification model...'):
    classifier = c_model()
    model = classifier.load_model(selected_model)


tab1, tab2 = st.tabs(["Single Comment", "Multiple Comment"])

with tab1:
    st.subheader('Single comment classification')
    text_input = st.text_area(label='Paste your text below (max 256 words)',
                              value='Hiiiiiiiii')
    MAX_WORDS = 256
    res = len(re.findall(r"\w+", text_input))
    if res > MAX_WORDS:
        st.warning(
            "⚠️ Your text contains "
            + str(res)
            + " words."
            + " Only the first 256 words will be reviewed! 😊"
        )
        text_input = text_input[:MAX_WORDS]

    submit_button = st.button(label='Submit comment')
    if submit_button:
        with st.spinner('Predicting ...'):
            start_time = time.time()
            time.sleep(2)
            preds, preds_prob = classifier.predict(text_input)
            end_time = time.time()
            p_time = round(end_time-start_time, 2)
            st.success(f'Prediction finished in {p_time}s!')

            label2id = {0: 'Positive', 1: 'Negative'}
            st.write(f'Label: {label2id[preds]}, with certainty: {preds_prob}')


with tab2:
    st.subheader('Multiple comment classification')
    file_input = st.file_uploader(label='Choose a file:', type='csv')
    if file_input:
        try:
            df = pd.read_csv(file_input)
        except:
            st.write('Bad File Error...')

        st.write(f"First 5 rows of {file_input.name}")
        st.write(df.head())

        submit_button = st.button(label='Submit file')
        if submit_button:
            with st.spinner('Predicting ...'):
                start_time = time.time()
                time.sleep(2)
                preds, preds_prob = classifier.predict(text_input)
                end_time = time.time()
                p_time = round(end_time-start_time, 2)
                st.success(f'Prediction finished in {p_time}s!')

                label2id = {0: 'Positive', 1: 'Negative'}
                flag = True
                # st.write(f'Label: {label2id[preds]}, with certainty: {preds_prob}')
                # st.write(f'Prob: {preds_prob}')

            results = [["keybert library", 0.4704, 1],
                    ["keyword extractor", 0.4535, 1],
                    ["extractor app", 0.4334, 1],
                    ["bert keyword", 0.4125, 1],
                    ["amazing keybert", 0.4066, 1],
                    ["app is", 0.3821, 1],
                    ["interface built", 0.3249, 1],
                    ["easy to", 0.3171, 1],
                    ["streamlit for", 0.3087, 1],
                    ["maarten grootendorst", 0.2766, 1]]


            c1, c2 = st.columns([3, 1])
            with c1:
                st.subheader("🎈 Check & download results")
            with c2:
                CSVButton2 = download_button(results, "Data.csv", "📥 Download (.csv)")
            flag = True

            st.header("")

            df = (
                DataFrame(results, columns=["Text", "Prob", "Label"])
                .sort_values(by="Prob", ascending=False)
                .reset_index(drop=True)
            )
            df.index += 1

            import seaborn as sns
            # Add styling
            cmGreen = sns.light_palette("green", as_cmap=True)
            cmRed = sns.light_palette("red", as_cmap=True)
            df = df.style.background_gradient(
                cmap=cmGreen,
                subset=["Prob"],
            )

            df = df.format({"Prob": "{:.2}"})
            st.table(df)





# with st.form(key="my_form"):
#
#
#     ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
#     with c1:
#         ModelType = st.radio(
#             "Choose your model",
#             ['XLM-RoBERTa', 'ParsBERT', 'ALBERT'],
#             help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text.",
#         )
#
#         Diversity = st.slider(
#             "Keyword diversity (MMR only)",
#             value=0.5,
#             min_value=0.0,
#             max_value=1.0,
#             step=0.1,
#             help="""The higher the setting, the more diverse the keywords.
#
#             Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.
#             """,
#         )
#
#     with c2:
#         doc = st.text_area(
#             "Paste your text below (max 256 words)",
#         )
#         submit_button = st.form_submit_button(label="✨ Get me the data!")

