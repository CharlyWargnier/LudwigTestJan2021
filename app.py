import streamlit as st
from ludwig.api import LudwigModel
#import numpy as np
import io
import pandas as pd
import json



#import pandas as pd
#import pandas as pd
#import streamlit as st
#from pyecharts import options as opts
#from pyecharts.charts import Tree
#from streamlit_echarts import st_echarts


#import requests
#import base64
#import time
#import os


st.title('Ludwig App')

##################################################

#region Layout size ####################################################################################

def _max_width_():
    max_width_str = f"max-width: 1600px;"
    #max_width_str = f"max-width: 1550px;"
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

'''
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/CharlyWargnier/CSVHub/main/CK-Tommy-All/Tommy-Macys-encoding-ISO-8859-1/Ludwig%20Files%20-December%202020/TH_tagged_deduped_dec_2020.csv",names=header_list)
df.info()
'''

st.stop()
##############################################


st.header('Tommy-Macys-encoding')
df = pd.read_csv("https://raw.githubusercontent.com/CharlyWargnier/CSVHub/main/CK-Tommy-All/Tommy-Macys-encoding-ISO-8859-1/Ludwig%20Files%20-December%202020/TH_tagged_deduped_dec_2020.csv")
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









