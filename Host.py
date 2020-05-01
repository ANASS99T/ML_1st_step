from flask import Flask, request
import json
import string
from nltk.stem import PorterStemmer
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
punctuations = string.punctuation

Stopwords = {''}
Stopwords_fr = set(stopwords.words('french'))
Stopwords_eng = set(stopwords.words('english'))
add_stop_words = "all almost also any because cause could do does done don't end for never see soon sorry than ourselves hers between yourself ut again there about once during out very having with they own an be some for do its yours such into of most itself other off is s am for who as from him each the themselves until below are we these your his through don nor me were her more himself this down should our their while above both up to ours had she all no when at any before them same and been have in will on does yourselves then that because what over why so can did not now under he you herself has just where too only myself which those i after few whom t being if theirs my against a by doing it how further was here than a à â abord afin ah ai aie ainsi allaient allo allô allons après assez attendu au aucun aucune aujourd aujourd'hui auquel aura auront aussi autre autres aux auxquelles auxquels avaient avais avait avant avec avoir ayant b bah beaucoup bien bigre boum bravo brrr c ça car ce ceci cela celle celle-ci celle-là celles celles-ci celles-là celui celui-ci celui-là cent cependant certain certaine certaines certains certes ces cet cette ceux ceux-ci ceux-là chacun chaque cher chère chères chers chez chiche chut ci cinq cinquantaine cinquante cinquantième cinquième clac clic combien comme comment compris concernant contre couic crac d da dans de debout dedans dehors delà depuis derrière des dès désormais desquelles desquels dessous dessus deux deuxième deuxièmement devant devers devra différent différente différentes différents dire divers diverse diverses dix dix-huit dixième dix-neuf dix-sept doit doivent donc dont douze douzième dring du duquel durant e effet eh elle elle-même elles elles-mêmes en encore entre envers environ es ès est et etant étaient étais était étant etc été etre être eu euh eux eux-mêmes excepté f façon fais faisaient faisant fait feront fi flac floc font g gens h ha hé hein hélas hem hep hi ho holà hop hormis hors hou houp hue hui huit huitième hum hurrah i il ils importe j je jusqu jusque k l la là laquelle las le lequel les lès lesquelles lesquels leur leurs longtemps lorsque lui lui-même m ma maint mais malgré me même mêmes merci mes mien mienne miennes miens mille mince moi moi-même moins mon moyennant n na ne néanmoins neuf neuvième ni nombreuses nombreux non nos notre nôtre nôtres nous nous-mêmes nul o o| ô oh ohé olé ollé on ont onze onzième ore ou où ouf ouias oust ouste outre p paf pan par parmi partant particulier particulière particulièrement pas passé pendant personne peu peut peuvent peux pff pfft pfut pif plein plouf plus plusieurs plutôt pouah pour pourquoi premier première premièrement près proche psitt puisque q qu quand quant quanta quant-à-soi quarante quatorze quatre quatre-vingt quatrième quatrièmement que quel quelconque quelle quelles quelque quelques quelqu'un quels qui quiconque quinze quoi quoique r revoici revoilà rien s sa sacrebleu sans sapristi sauf se seize selon sept septième sera seront ses si sien sienne siennes siens sinon six sixième soi soi-même soit soixante son sont sous stop suis suivant sur surtout t ta tac tant te té tel telle tellement telles tels tenant tes tic tien tienne tiennes tiens toc toi toi-même ton touchant toujours tous tout toute toutes treize trente très trois troisième troisièmement trop tsoin tsouin tu u un une unes uns v va vais vas vé vers via vif vifs vingt vivat vive vives vlan voici voilà vont vos votre vôtre vôtres vous vous-mêmes vu w x y z zut"
add_stop_words = add_stop_words.split()
Stopwords.update(add_stop_words)
Stopwords.update(Stopwords_fr)
Stopwords.update(Stopwords_eng)

def stemSentence(sentence):
    token_words=word_tokenize(str(sentence))
    token_words
    porter = PorterStemmer()
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def funct(text):
    text = str(text).lower()
    
    translator = str.maketrans(punctuations," "*len(punctuations))
    s = text.translate(translator)
    
    res = ''.join([i for i in s if not i.isdigit()])
    
    wordtokens = word_tokenize(res)
    
    return  ' '.join([w for w in wordtokens if not w in Stopwords])

app = Flask(__name__)
@app.route('/', methods = ['POST'])
def index():
    data = dict()
    if request.method == 'POST' and 'dsc' in request.form:
        dsc = request.form['dsc']
        #print("\n\n\n\n\n\n dsc ==> ", type(dsc),"\n\n\n\n\n")
        new_dsc = stemSentence(dsc)
        new_dsc2 = funct(new_dsc)

        #load tfidf
        tfidf_vectorizer = pickle.load(open("tfidf.pkl", "rb"))
        
        var = tfidf_vectorizer.transform([new_dsc2])
        #load model
        my_model = 'final_model.pkl'
        with open(my_model, 'rb') as f:
            model = pickle.load(f)
        
        data['prediction'] = str(model.predict(var)[0])
        data['text'] = str(new_dsc2)
        json_data = json.dumps(data)
        return json_data
    else:
        data['error'] = "no response"
        return json_data
    
if __name__ == '__main__':
    app.run(debug=True)
        
