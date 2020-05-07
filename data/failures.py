from gensim.corpora.wikicorpus import filter_wiki

example = '{{veröffentlicht|04:33, 20. Apr. 2013 (CEST)}\n{{Beginn|Berlin|Deutschland|20.04.2013}} [[:Kategorie:19.04.2013|Gestern]] löste ein Brief für Bundespräsident [[w:Joachim Gauck|Joachim Gauck]] bei einer Durchleuchtung der Poststelle des Bundespräsidialamts Alarm aus: Man vermutete Sprengstoff im Brief. Daraufhin wurde der Brief im Park von Schloss Bellevue von Experten einer Sprengung ausgesetzt. Nach der Sprengung erwies sich das Ganze jedoch als Fehlalarm, Sprengstoff war offenbar nicht enthalten. Gauck selbst hielt sich zu der Zeit nicht in Bellevue auf. \n\nAnfang der Woche und zeitnah zum [[Boston-Marathon-Anschlag: Tatverdächtiger erschossen|Anschlag auf den Boston-Marathon]] waren an US-Präsident Barack Obama sowie einen US-Senator Giftbriefe geschickt worden.\n\n{{Kommentieren}}\n\n\n\n'

print('Text before filter_wiki from gesim :')
print([example])
print()
print('---------------------------------')
print('Text after filter_wiki from gesim :')
print([filter_wiki(example)])