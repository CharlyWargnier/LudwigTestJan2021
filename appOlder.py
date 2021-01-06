import streamlit as st
from ludwig.api import LudwigModel
#import numpy as np
import io
import pandas as pd
import json
import os
cwd = os.getcwd()


st.markdown("---")

st.title('Ludwig App - train + load models ')

with st.beta_expander("📝 - Roadmap ⯈", expanded=True):
  st.write("""   
- Add 'IF Function'-> if files are downloaded on CWD, then load directly
- Try to cache training model (might not be essential as files will be saved)
- Cache final table
- Then add slider to filter accuracy
 
 """)


with st.beta_expander(" Done ", expanded=True):
  st.write("""   
- Try to load saved file on CWD on S4
- Get the training to load on the uploaded CSV file 
- Add auto-rename on columns from other file!

 """)


st.markdown("---")

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


#Reuters
#df = pd.read_csv("https://raw.githubusercontent.com/CharlyWargnier/CSVHub/main/FromLudwigsWebsite/reuters-allcats.csv")

#df = pd.read_csv("https://raw.githubusercontent.com/CharlyWargnier/CSVHub/main/CK-Tommy-All/Tommy-Macys-encoding-ISO-8859-1/Ludwig%20Files%20-December%202020/THKeywordsTrainingSet300.csv")
#Reuters
#df = pd.read_csv("https://raw.githubusercontent.com/CharlyWargnier/CSVHub/main/FromLudwigsWebsite/reuters-allcats.csv")

#st.stop()

##################################################

#region Layout size ####################################################################################

def _max_width_():
    max_width_str = f"max-width: 1600px;"
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

c1, c2 = st.beta_columns(2)

with c1:

    multiple_files = st.file_uploader(
        "Multiple File Uploader",
        accept_multiple_files=True
    )
    for file in multiple_files:
        file_container = st.beta_expander(
            f"File name: {file.name} ({file.size})"
        )
        data = io.BytesIO(file.getbuffer())
        file_container.write(pd.read_csv(data, error_bad_lines=False))
        file.seek(0)

    dfs = [pd.read_csv(file) for file in multiple_files]
    #dfs = [pd.read_csv(data, header=None, delim_whitespace=True) for file in multiple_files]

with c2:
    st.write('')
    st.write('')
    st.success('file 1 uploaded')
    st.success('file 2 uploaded')
    st.success('file 3 uploaded')

############################################

#st.header("list index is set to 2 files")
#st.write("- Print list of dataframes")
#st.write(dfs)

if not dfs:
    st.warning('waiting for dataframes to be loaded')
    st.stop()

st.write("dfs[0]")
df = dfs[0]
df.columns = ["doc_text", "class"]
df

st.header('Files! Tommy-Macys or Reuters')
#Tommy-Macys-10K
#df = pd.read_csv("https://raw.githubusercontent.com/CharlyWargnier/CSVHub/main/CK-Tommy-All/Tommy-Macys-encoding-ISO-8859-1/Ludwig%20Files%20-December%202020/TH_tagged_deduped_dec_2020.csv")

#Tommy-Macys-300
#df = pd.read_csv("https://raw.githubusercontent.com/CharlyWargnier/CSVHub/main/CK-Tommy-All/Tommy-Macys-encoding-ISO-8859-1/Ludwig%20Files%20-December%202020/THKeywordsTrainingSet300.csv")

#Reuters
#df = pd.read_csv("https://raw.githubusercontent.com/CharlyWargnier/CSVHub/main/FromLudwigsWebsite/reuters-allcats.csv")

df


input_features =  [{'name': 'doc_text', 'type': 'text'}]
output_features = [{'name': 'class', 'type': 'category'}]


config = {
    'input_features': input_features,
    'output_features': output_features,
    'combiner': {'type': 'concat', 'fc_size': 14},
    'training': {'epochs': 2}
}


model = LudwigModel(config)

train_stats, _, _ = model.train(dataset=df)

st.header('Eval Stats')
eval_stats, _, _ = model.evaluate(dataset=df)
#st.write(eval_stats)
#st.write(type(eval_stats))

#WORKS!
st.subheader('In JSON format')
json_object = json.dumps(str(eval_stats), indent = 4)   
st.write(json_object)  
#DOESN'T WORK YET
st.subheader('In dataframe format')
st.write('separate dictionnaries from main dictionnary')
df = pd.DataFrame([eval_stats], columns=eval_stats.keys())
st.table(df)


######################
#Save model

#Works
#model.save('C:/Users/Charly/Desktop/LudwigNew/')

#Works
model.save(cwd)

#Load model
#modelLoaded = LudwigModel.load('C:/Users/Charly/Desktop/LudwigNew/')
modelLoaded = LudwigModel.load(cwd)


# New set, deduped
#df2 = pd.read_csv("https://raw.githubusercontent.com/CharlyWargnier/CSVHub/main/CK-Tommy-All/Tommy-Macys-encoding-ISO-8859-1/For%20Ludwig/TH_NOT_tagged_deduped_dec_2020.csv")
df2 = pd.read_csv("https://raw.githubusercontent.com/CharlyWargnier/CSVHub/main/CK-Tommy-All/Tommy-Macys-encoding-ISO-8859-1/Ludwig%20Files%20-December%202020/TH_NOT_tagged_deduped_dec_2020.csv")
#df2.head()
df2

# Predict on new dataset
predictions, _ = modelLoaded.predict(dataset=df2)
#predictions

# ETL on predicted dataset
## Removed unsued columns
predictionsNew = predictions[['class_predictions', 'class_probability']]
#predictionsNew.head(3)
predictionsNew

## Merge df2 and predictions on indices
dfFinal2 = pd.merge(predictionsNew, df2, left_index=True, right_index=True)
#dfFinal2.head(2)
st.table(dfFinal2)


