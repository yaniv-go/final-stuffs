from google_images_search import GoogleImagesSearch

def my_progressbar(url, progress):
    print(url + ' ' + progress + '%')

gis = GoogleImagesSearch('AIzaSyCyRgplN2X_vyGUna3A3KKdnvC-Il_5aJA', 'dd3d19e417385eecd', progressbar_fn=my_progressbar)

_search_params = {
    'q': 'big cats',
    'num': 10
}
gis.search(search_params=_search_params)

print(gis.results())
