import streamlit as st
from ludwig.api import LudwigModel
#import numpy as np
import io
import pandas as pd
import json
import os
cwd = os.getcwd()


#cwd = os.getcwd()
cwd

[x[0] for x in os.walk(cwd)]



#import streamlit as st
from streamlit.hashing import _CodeHasher

#region Layout size ####################################################################################

def _max_width_():
    max_width_str = f"max-width: 1300px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()

#endregion Layout size ####################################################################################


try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server

def main():
    state = _get_state()
    pages = {
        "Train Model": page_settings,
        "Make predictions!": page_dashboard,
    }


    st.sidebar.title(":floppy_disk: Page states")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


def page_dashboard(state):
    st.title(":chart_with_upwards_trend: Make Predictions!")
    
    import pandas as pd
    import io
    import base64

    import os
    #ModelFiles = os.path.isfile("test.csv")
    ModelFiles = os.path.isfile("training_set_metadata.json")

    #st.markdown("---")

    #st.title('Ludwig App - train + load models ')

    with st.beta_expander("📝 - Roadmap ⯈", expanded=True):
        st.write("""   
        - Deploy app on S4
        - PREDICT TAB!  PREDICT TAB!  PREDICT TAB!  PREDICT TAB!
        - Last table -> Add colour formatting on accuracy metrics
        - TRAINING TAB!  TRAINING TAB!  TRAINING TAB!  TRAINING TAB!
        - Reformat accuracy results - Make it more legible
        - LATER/2DARY - LATER/2DARY - LATER/2DARY - LATER/2DARY - LATER/2DARY - LATER/2DARY!!!
        - Finish Transfering code and notes to 'stateSynodeWorking2021'
        - Find name + logo
        - Add gif during training
        - Decide on emoji to use: 🧠 💪 - "Train your text classifier! ")
        - DOESN'T SEEM TO WORK DOESN'T SEEM TO WORK!!!
        - Try to cache training model (might not be essential as files will be saved)
        """)

    with st.beta_expander(" Done ", expanded=False):
        st.write("""   
        - Add an arrow to show ppl they have to click on the other tab        
        - PREDICT TAB!  PREDICT TAB!  PREDICT TAB!  PREDICT TAB!
        - Last table -> Change columns' order
        - Add an export module
        - OTHER  OTHER  OTHER  OTHER
        - Add tick box with the word "model has been trained!"
        - Remove cache from api.py (ludwig file) as doesn't seem to be working
        - Fixed 'EmptyDataError: No columns to parse from file ' via uploaded_file.seek(0).
        - Change Multiple File Uploader to single file uploader
        - Add single file uploader for model loading (not training!) phase
        - Add tabbed navigation via Synode's template
        - Add 'IF Function'-> if files are downloaded on CWD, then load directly
        - Try to load saved file on CWD on S4
        - Get the training to load on the uploaded CSV file 
        - Add auto-rename on columns from other file!

        """)
    
    with st.beta_expander(" Later ", expanded=False):
        st.write("""   
        - PREDICT TAB!  PREDICT TAB!  PREDICT TAB!  PREDICT TAB!
        - Add textArea
        - Then add slider to filter accuracy
        - TRAINING TAB!  TRAINING TAB!  TRAINING TAB!  TRAINING TAB!
        - Add how many keywords have been reviewd
        - Add how many labels have been reviewed
    
        """)

    st.header('Files are located: "Desktop\LudwigNew\CSVFilesToUpload"')

    c1, c2, c3 = st.beta_columns(3)

    with c1:
        st.write('TH_tagged_deduped_dec_2020')
        st.write('https://bit.ly/2KZEtZI')

    with c2:
        st.write('Tommy-Macys-300')
        st.write('https://bit.ly/3b4cJhu')

    with c3:
        st.write('Reuters')
        st.write('https://bit.ly/3o9Zqzr')

    if not ModelFiles:
        st.warning('Please train a model first!')
        #st.stop()
        #st.success('ModelFiles_are_saved')
     
    
    uploaded_file = st.file_uploader("Choose a file", key = 1)
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dfPredictions = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        st.write(dfPredictions)
        

    else:
        st.success('Models are uploaded, please upload a CSV!')
        st.stop()

    modelLoaded = LudwigModel.load(cwd)

    st.write('TH_NOT_tagged')
    st.write('https://bit.ly/3hSmMYk')

    #uploaded_file = st.file_uploader("Choose a file")  
    #if uploaded_file is not None:
    #    # Can be used wherever a "file-like" object is accepted:
    #    dfPredictions = pd.read_csv(uploaded_file)
    #    st.write(dfPredictions)
    #else:
    #    st.success('Please upload a dataset and get instant predictions!')
    #    st.stop()

    
    dfPredictions

    # Predict on new dataset
    predictions, _ = modelLoaded.predict(dataset=dfPredictions)
    #predictions

    # ETL on predicted dataset
    ## Removed unsued columns
    predictionsNew = predictions[['class_predictions', 'class_probability']]
    #predictionsNew.head(3)
    predictionsNew

    ## Merge dfPredictions and predictions on indices
    dfFinal2 = pd.merge(predictionsNew, dfPredictions, left_index=True, right_index=True)
    #dfFinal2.head(2)
    
    dfFinal2.rename({'doc_text': 'Keyword', 'class_predictions': 'Predicted label', 'class_probability': 'Probability (in %)'  }, axis=1, inplace=True)
    column_names = ["Keyword", "Predicted label", "Probability (in %)"]
    dfFinal2 = dfFinal2.reindex(columns=column_names)
    
    st.header("")
    c = st.beta_container()   
    st.table(dfFinal2)

    try:
        csv = dfFinal2.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        c.markdown('## **③ Check results or download CSV **')
        c.subheader("")
        href = f'<a href="data:file/csv;base64,{b64}" download="filtered_table.csv">** ⯈ Download link 🎁 **</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    except NameError:
        print ('wait')

    display_state_values(state)


def page_settings(state):
    st.title("Train your text classifier!")
    display_state_values(state)

    #st.write("---")  

    import os
    #ModelFiles = os.path.isfile("test.csv")
    ModelFiles = os.path.isfile("training_set_metadata.json")

    if not ModelFiles:

        st.warning('Please train a model first!')
        #st.stop()
        #st.success('ModelFiles_are_saved')
    
        import pandas as pd
        import io
        import base64

        uploaded_file = st.file_uploader("Choose a file", key = 2)

        if uploaded_file is not None:
            # Can be used wherever a "file-like" object is accepted:
            df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            #df.seek(0)
            df.columns = ["doc_text", "class"]
            st.write(df)

        else:
            #st.warning('Upload the CSV to be trained')
            st.stop()
           
        input_features =  [{'name': 'doc_text', 'type': 'text'}]
        output_features = [{'name': 'class', 'type': 'category'}]

        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
            'training': {'epochs': 2}
        }

        model = LudwigModel(config)

        import pandas as pd

        train_stats, _, _ = model.train(dataset=df)

        st.header('Eval Stats')
        eval_stats, _, _ = model.evaluate(dataset=df)
        #st.write(eval_stats)
        #st.write(type(eval_stats))

        #WORKS!
        st.subheader('In JSON format')
        json_object = json.dumps(str(eval_stats), indent = 4)   
        #DOESN'T WORK YET
        st.subheader('In dataframe format')
        st.write(json_object)  
        st.write('separate dictionnaries from main dictionnary')
        df = pd.DataFrame([eval_stats], columns=eval_stats.keys())
        #st.table("df")
        #st.table(df)

        #Save model
        model.save(cwd)

    else:
        st.success('✅ The model has now been trained, you can start making predictions!')
        st.image('arrow2.png', width = 325)

        #json_object = json.dumps(str(eval_stats), indent = 4)   
        #st.write(json_object)  

    #csvLeft = df.to_csv()
    #b642 = base64.b64encode(csvLeft.encode()).decode()
    #href = f'<a href="data:file/csvLeft;base64,{b642}" download="EntitiesIn02Not01.csv">** ⯈ Download entities in URL #02 not #01.**</a>'
    #st.markdown(href, unsafe_allow_html=True)


    #options = ["Hello", "World", "Goodbye"]
    #state.input = st.text_input("Set input value.", state.input or "")
    #state.slider = st.slider("Set slider value.", 1, 10, state.slider)
    #state.radio = st.radio("Set radio value.", options, options.index(state.radio) if state.radio else 0)
    #state.checkbox = st.checkbox("Set checkbox value.", state.checkbox)
    #state.selectbox = st.selectbox("Select value.", options, options.index(state.selectbox) if state.selectbox else 0)
    #state.multiselect = st.multiselect("Select value(s).", options, state.multiselect)
#
    ## Dynamic state assignments
    #for i in range(3):
    #    key = f"State value {i}"
    #    state[key] = st.slider(f"Set value {i}", 1, 10, state[key])


def display_state_values(state):
    st.write("")
    #st.write("Input state:", state.input)
    #st.write("Slider state:", state.slider)
    #st.write("Radio state:", state.radio)
    #st.write("Checkbox state:", state.checkbox)
    #st.write("Selectbox state:", state.selectbox)
    #st.write("Multiselect state:", state.multiselect)
    #
    #for i in range(3):
    #    st.write(f"Value {i}:", state[f"State value {i}"])
    #
    #if st.button("Clear state"):
    #    state.clear()


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()