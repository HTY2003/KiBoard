import json
from wordfreq import zipf_frequency, top_n_list

jsondict = {word:zipf_frequency(word, "en") for word in top_n_list('en', 20000)}

with open('english_zipf.json', 'w') as outfile:
    json.dump(jsondict, outfile)
