# Problem & Solution with NLTK data packages in Docker

NLTK functions/classes like "word_tokenize", "stopwords", "PorterStemmer" and "WordNetLemmatizer", are dependent on language datafiles.

These were nmot automatically downloaded when importing NLTK, when creating the Docker image.

## Solution
Found on: https://stackoverflow.com/questions/31143015/docker-nltk-download

First tip on the stack-thread was implemented first.
- it worked when running the image/container on local machine
  - it consist of running the nltk download functions wehn creating the image (`RUN ...`)
- but, when uploading and running on GCP, the old errors came back

Further down the thread is a more **"hard coded"** solution
- here we make a copy of all the needed nltk files here in the project directory, in this `nltk-data` folder
- then `COPY` this folder, alongside with the other needed project files and folders, when creating the image.
- 👍 **it works** 😁
