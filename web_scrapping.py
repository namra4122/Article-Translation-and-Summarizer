# **WEB SCRAPPING**
import requests
from bs4 import BeautifulSoup

r = requests.get('https://www.divyabhaskar.co.in/utility/automobile/news/this-suv-will-also-run-on-e-20-fuel-the-information-was-leaked-before-the-launch-131086628.html')
 
# check status code for response received
# success code - 200
print(r)
 
# print content of request
print(r.content)

soup = BeautifulSoup(r.content, 'html.parser')
titlesHTML=soup.find_all('h1')
content = soup.find_all('p')
 
print(titlesHTML)
print(content)

titles=[]
for line in titlesHTML:
  titles.append(line.text)
print(titles)

text=[]

for line in content:
  text.append(line.text)

print(text)