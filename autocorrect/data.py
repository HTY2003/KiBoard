import json
from wordfreq import word_frequency, top_n_list

jsondict = {word:word_frequency(word, "en") for word in top_n_list('en', 15000)}

with open('english.json', 'w') as outfile:  
    json.dump(jsondict, outfile)