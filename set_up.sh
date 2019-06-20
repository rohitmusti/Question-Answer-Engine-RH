# sets up all of the installs 
echo "#################################"
echo "## setting up package installs ##"
echo "#################################"
pip3 install pandas --user   
pip3 install numpy --user
pip3 install torch --user
pip3 install ujson --user
pip3 install tensorflow --user
pip3 install urllib3 --user
pip3 install spacy --user
pip3 install tqdm --user
pip3 install nltk --user
pip3 install spacy --user
python3 -m spacy download en --user

# creates directories
echo "#################################"
echo "## setting up data directories ##"
echo "#################################"
mkdir data/train -p
mkdir data/dev -p
mkdir data/test -p
mkdir data/embeddings -p
mkdir logs

# downloading data
echo "##########################"
echo "## downloading the data ##"
echo "##########################"
wget --directory-prefix=data/train/ https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget --directory-prefix=data/dev/ https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
wget --directory-prefix=data/embeddings/ http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip data/embeddings/glove.840B.300d.zip -d data/embeddings/


