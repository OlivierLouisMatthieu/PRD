import urllib

# # open a connection to a URL using urllib
# webUrl = urllib.request.urlopen("https://ucafr-my.sharepoint.com/personal.html")
# # get the result code and print it
# print("result code: ", webUrl)
#
# # read the data from the URL and print it
# data = webUrl.read()
# print(data)

# Assign the open file to a variable
webFile = urllib.urlopen("https://askcodez.com/python-lire-le-fichier-a-partir-dune-url-de-site-web.html")

# Read the file contents to a variable
file_contents = webFile.read()
print(file_contents)

# set up authentication info
# authinfo = urllib.request.HTTPBasicAuthHandler()
# authinfo.add_password(realm='PDQ Application',
#                       uri='https://mahler:8092/site-updates.py',
#                       user='stanislas.malfait@etu.uca.fr',
#                       passwd='geheim$parole')

# proxy_support = urllib.request.ProxyHandler({"https" : "https://ucafr-my.sharepoint.com/personal/stanislas_malfait_etu_uca_fr"})
#
# # build a new opener that adds authentication and caching FTP handlers
# opener = urllib.request.build_opener(proxy_support, authinfo,
#                                      urllib.request.CacheFTPHandler)




