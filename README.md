# cs5293sp23-project3
# Text analytics - The Smart City Slicker
## author @Vasu Deva Sai Nadha Reddy Janapala

**Key points**:\
Initially, a model was developed using Jupyter Notebook in this project.\
Subsequently, this model was utilized in the .py file to make predictions for the new .pdf file.\
The output of this process is a .tsv file.\

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
Experimenting with clustering models. Tried different models. The following is the samepl expected table.
