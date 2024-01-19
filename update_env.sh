pip install -U scikit-learn
pip install matplotlib
pip install wandb
pip install tokenizers==0.11.6
pip install transformers==4.18.0
pip install setuptools==65.5.0
pip install gym==0.21.0
pip install stable-baselines3==1.5.1a5
pip install jsonlines
pip install spacy
pip install bert-score
pip install datasets
pip install nltk
pip install pandas
pip install rich
pip install shimmy
pip install tqdm
pip install rouge_score
pip install sacrebleu
pip install absl-py
pip install six
pip install logzero
pip install pycountry
pip install sacremoses
pip install diskcache
pip install sentencepiece
pip install tabulate
pip install seaborn
pip install protobuf==3.20.3
pip install py-rouge==1.1
pip install "importlib-metadata<5.0"

# install java if not already installed
#add-apt-repository ppa:openjdk-r/ppa
#apt-get install -y openjdk-8-jdk
#apt-get install -y openjdk-8-jre
#update-alternatives --config java
#update-alternatives --config javac

# download external models (since it requires dependencies)
pip install markupsafe==2.0.1
python -c "import nltk; nltk.download('punkt')"
python -m spacy download en_core_web_sm
