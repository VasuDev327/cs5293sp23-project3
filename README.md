# cs5293sp23-project3
# Text analytics - The Smart City Slicker
## author @Vasu Deva Sai Nadha Reddy Janapala

**Key points**:\
Initially, a model was developed using Jupyter Notebook in this project.\
Subsequently, this model was utilized in the .py file to make predictions for the new .pdf file.\
The output of this process is a .tsv file.\

### Used libraries
```
import os
from pypdf import PdfReader
import re 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import argparse
from pypdf import PdfReader
import os
import re
import pandas as pd
import nltk
import spacy
import unicodedata
import re
from nltk.corpus import wordnet
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
```


### Step 1:
To start the project, all the 71 files available in the instructor's Git repository were downloaded, which included both .doc and .pdf files.\
However, only the pdf files were considered for this project.\
Once the raw text was extracted from these pdf files, the city and state names were extracted from the file names.\
This information was then stored in a data frame, as illustrated below.

![image](https://user-images.githubusercontent.com/102677891/236913049-03a638ec-9bc4-4dba-834c-d9584b87695e.png)

### Step 2:
To clean and store the extracted raw data in a dataframe, existing Python files like text_normalizer.py and contractions.py from the textbook "Text Analytics with Python" were used for preprocessing the data.\
However, there is a drawback to this approach which will be discussed under the bugs section.\
After performing normalization, attempts were made to apply Ngrams to the data, but it did not seem to have any effect on the dataframe, and as a result, it was ignored.

### Step 3:
Explored various clustering models and tested them with the data. Below is a sample of the graph and expected table.
![image](https://user-images.githubusercontent.com/102677891/236914745-0a2d6097-5d35-4c0a-bf5e-a4c8f7b83dec.png)\
![image](https://user-images.githubusercontent.com/102677891/236914789-3029a954-9b40-4a6e-b920-ee2f96ca19eb.png)\
![image](https://user-images.githubusercontent.com/102677891/236914840-578d59c7-66d1-4a8f-ad87-a1f2180ed62b.png)
### Step 4:
The cluster ID has been saved based on the chosen value of k for the clustering model. For this project, k was chosen to be 2 based on the elbow method.

### Step 5:
Here the trained model has been saved in a pickle form, which is used in the project3.py file when a new city.pdf file is reading.
Then the themes are explored. Also, stored in the pickle format, to further use it in the proejct3.py.
Thereby TOPIC IDS are added.

Now, the required data is stored in the .tsv file, the output of it looks as follows:\
![image](https://user-images.githubusercontent.com/102677891/236915893-929ce827-b4ad-4bd8-8d95-54f194feb56c.png)

### Step 6:
The args will be passed
![image](https://user-images.githubusercontent.com/102677891/236917702-eb488987-9f7d-4dd3-b1bf-089a1416e778.png)

### Step 7:
The same preprocessing has to happen in the project3.py when a new city.pdf file is read, post that the saved model is used. Which is shown as below:
```
with open('model.pkl', 'rb') as file:
    loaded_model, loaded_vectorizer = pickle.load(file)
with open('LDA model.pkl', 'rb') as f:
    lda_loaded, vectorizer_loaded_l = pickle.load(f)
```
Through these the new data will be predicted. So the predicted data will generate the output as follows and also saves the complete format in the .tsv file format.
## Output:
![image](https://user-images.githubusercontent.com/102677891/236917039-845a3b92-30e0-4814-9c72-55c519872287.png)


