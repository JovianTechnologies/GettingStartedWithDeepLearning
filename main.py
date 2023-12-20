from dotenv import load_dotenv
import fastbook
from fastbook import *
from fastai.vision.widgets import *

load_dotenv()
azure_key = os.environ.get('AZURE_KEY')

def test_azure_key():
    results = search_images_bing(azure_key, 'mushroom', max_images=1)
    ims = results.attrgot('contentUrl')

    # print length of results to console
    print(len(ims))

    d = 'mushroom.jpg'
    download_url(ims[0], d)
    image = Image.open(d)
    resized_image = image.to_thumb(128, 128)
    resized_image.show()

def main():
    test_azure_key()


if __name__ == "__main__":
    main()
