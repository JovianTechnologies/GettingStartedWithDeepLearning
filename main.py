import fastbook
from fastai.vision.widgets import ImageClassifierCleaner
from fastbook import *
import matplotlib.pyplot as pl

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

def train_model(path):
    mushrooms = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128))

    mushrooms = mushrooms.new(
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms())

    dls = mushrooms.dataloaders(path, num_workers=0) # num_workers=0 to avoid a warning on windows

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)

    return learn



def examine_model(model):
    interp = ClassificationInterpretation.from_learner(model)
    interp.plot_confusion_matrix()

    time.sleep(10)

    interp.plot_top_losses(5, nrows=1)



def main():
    # mushroom_types = 'jack o\' lantern', 'chanterelle'
    # get_images('images/mushroom', mushroom_types)
    # remove_png_files('images/chanterelle')
    # remove_png_files('images/jack o\' lantern')

    model = train_model('images')
    examine_model(model)


if __name__ == "__main__":
    main()