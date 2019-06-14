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

# creates directories
echo "#################################"
echo "## setting up data directories ##"
echo "#################################"
mkdir data/train -p
mkdir data/dev -p
mkdir data/test -p
mkdir data/embeddings -p
