# **TRANSLATION**

from googletrans import Translator, constants

translator = Translator()
for t in titles:
  translation = translator.translate(t)
  trans_title = translation.text

print(trans_title)

trans_text=[]

for t in text:
  translation = translator.translate(t)
  trans_text.append(translation.text)

text_final=",".join(trans_text)
print(text_final)