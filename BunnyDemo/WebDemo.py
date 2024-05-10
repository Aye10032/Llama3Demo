import streamlit as st

from BunnyDemo.LangchainDemo import BunnyLlama3

st.set_page_config(layout='wide')

col1, col2 = st.columns([1, 1], gap='large')


@st.cache_resource(show_spinner='loading llm...')
def get_llm():
    _llm = BunnyLlama3()

    return _llm


with col1:
    upload_file = st.file_uploader('Choose a picture', type=['jpg', 'png', 'jepg'], accept_multiple_files=False)
    st.text_area('Question', "Why is the image funny? Answer with chinese.", key="text_input")
    st.button("generate", type="primary", key="submit")

with col2:
    if upload_file is not None:
        with open(f'BunnyDemo/image/{upload_file.name}', 'wb') as f:
            f.write(upload_file.getbuffer())

        with st.chat_message('user'):
            st.image(f'image/{upload_file.name}', width=400)

    if st.session_state.get('submit'):
        prompt: str = st.session_state.get('text_input')
        with st.chat_message("user"):
            st.write(prompt)

        llm = get_llm()

        with st.spinner('generate answer...'):
            result = llm.invoke(prompt, img_url=f'image/{upload_file.name}')

        with st.chat_message('ai'):
            st.write(result)
