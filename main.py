import fastbook
from fastbook import *
from fastai.vision.widgets import *

azure_key = os.environ.get('AZURE_SEARCH_KEY', 'ef46f0c9d255472b95608666b8288a34')

def test_azure_key():
    results = search_images_bing(azure_key, 'grizzly bear')
    ims = results.attrgot('contentUrl')

    # print length of results to console
    print(len(ims))

    d = 'grizzly.jpg'
    download_url(ims[0], d)
    image = Image.open(d)
    resized_image = image.to_thumb(128, 128)
    resized_image.show()

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
    get_images('images/mushroom', mushroom_types)


if __name__ == "__main__":
    main()