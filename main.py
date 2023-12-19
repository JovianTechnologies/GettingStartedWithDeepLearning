from dotenv import load_dotenv
import fastbook
from fastbook import *
from fastai.vision.widgets import *

load_dotenv()
azure_key = os.environ.get('AZURE_KEY')
def test_azure_key():
    results = search_images_bing(azure_key, 'mushroom')
    ims = results.attrgot('contentUrl')

    # print length of results to console
    print(len(ims))

    d = 'mushroom.jpg'
    download_url(ims[0], d)
    image = Image.open(d)
    resized_image = image.to_thumb(128, 128)
    resized_image.show()

def remove_png_files(path_str):
    path = Path(path_str)
    for f in path.ls():
        if f.suffix == '.png':
            f.unlink()

def get_images(path_name, types):
    path = Path(path_name)
    if not path.exists():
        path.mkdir()

    for o in types:
        dest = (path / o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(azure_key, f'{o} {path_name}s')
        download_images(dest, urls=results.attrgot('contentUrl'))

    fns = get_image_files(path)
    print(fns)

    # remove any images that can't be opened
    failed = verify_images(fns)
    print(failed)
    failed.map(Path.unlink)


def main():
    mushroom_types = 'jack o\' lantern', 'chanterelle'

    get_images('images', mushroom_types)
    #remove any png files since they can have transparency
    remove_png_files('images/chanterelle')
    remove_png_files('images/jack o\' lantern')


if __name__ == "__main__":
    main()