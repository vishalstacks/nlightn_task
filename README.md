# nlightn_task

Hello Sir,
Welcome to the repo


Task:
Website dump
Create a python application as a library to download all the info on a website (static/dynamic). The input will be a website url and output might be json dump of all the pages navigable within the domain from the provided URL


To start:
Install all dependencies

- pip install aiohttp playwright beautifulsoup4 tldextract tqdm
- playwright install



run command: 
- python website_dumper.py https://example.com
  (or)
- python website_dumper.py https://example.com --max-pages 10 --max-depth 2 --output site_dump.json    (same command with more options)



Ouput:
you will see the output in site_dump.json file. 


optional:
to formate the json file to view nice and short use this command 
- python -m json.tool site_dump.json | less



I have attached a ouput site_dump.json file as well in the repo.

Thank you!
