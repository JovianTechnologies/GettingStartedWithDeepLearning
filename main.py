from dotenv import load_dotenv
from fastbook import *

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

def search_images_bing(key, term, min_sz=128, max_images=150, offset=0):
    params = dict(q=term, count=max_images, min_height=min_sz, min_width=min_sz, offset=offset)
    search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    response = requests.get(search_url, headers={"Ocp-Apim-Subscription-Key":key}, params=params)
    response.raise_for_status()
    return L(response.json()['value'])

def image_hash(image_path):
    with Image.open(image_path) as img:
        return hashlib.md5(img.tobytes()).hexdigest()

def remove_image_duplicates(dir_path):
    unique = {}
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            img_hash = image_hash(file_path)
            if img_hash not in unique:
                unique[img_hash] = filename
            else:
                print(f"Removing duplicate image: {filename}")
                os.remove(file_path)
    return unique

def get_images(path_name, types):
    path = Path(path_name)
    if not path.exists():
        path.mkdir()

    nOffsets = 3
    for o in types:
        dest = (path / o)
        dest.mkdir(exist_ok=True)
        results = []
        for offset in range(nOffsets):
            print(f'{o} offset: {offset}')
            results += search_images_bing(azure_key, f'{o} {path_name}s', max_images=150, offset=offset)

        download_images(dest, urls=results.attrgot('contentUrl'))

    fns = get_image_files(path)
    print(fns)

    # remove any images that can't be opened
    failed = verify_images(fns)
    print(failed)
    failed.map(Path.unlink)

def train_model(path):
    mushrooms = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms()
    )

    dls = mushrooms.dataloaders(path, num_workers=0) # num_workers=0 to avoid a warning on windows

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)

    return learn

def examine_model(model):
    interp = ClassificationInterpretation.from_learner(model)

    interp.plot_confusion_matrix()
    plt.show()

    interp.plot_top_losses(5, nrows=1)
    plt.show()



def main():
    # mushroom_types = {'Omphalotus olearius'}
    # mushroom_types = 'Omphalotus olearius', 'chanterelle'
    # get_images('images/', mushroom_types)

    remove_png_files('images/chanterelle')
    # remove_png_files('images/jack o\' lantern')
    remove_png_files('images/Omphalotus olearius')

    remove_image_duplicates('images/chanterelle')
    remove_image_duplicates('images/jack o\' lantern')
    # jack_o_lantern_path = Path('images/jack o\' lantern')
    # try:
    #     for f in jack_o_lantern_path.ls():
    #         f.unlink()
    #
    #     os.rmdir(jack_o_lantern_path)
    #     print(f"The directory {jack_o_lantern_path} has been deleted.")
    # except OSError as e:
    #     print(f"Error: {e.strerror}")

    model = train_model('images')
    examine_model(model)


if __name__ == "__main__":
    main()