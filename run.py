import os
import sys
from options.test_options import TestOneOptions
from models import create_model
import util.util as util
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run.py <first_path> <second_path>")
        sys.exit(1)

    opt = TestOneOptions().parse()  # get test options
    first_image_path, second_image_path = opt.paths
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = (
        True  # no flip; comment this line if results on flipped images are needed.
    )
    opt.display_id = (
        -1
    )  # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.parallelize()
    if opt.eval:
        model.eval()

    test_trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    x = Image.open(first_image_path).convert("RGB")
    x = test_trf(x)[None, ...]

    y = Image.open(second_image_path).convert("RGB")
    y = test_trf(y)[None, ...]

    result = model.forward_one(x, y)  # run inference
    result = util.tensor2im(result)

    name = lambda x: os.path.splitext(os.path.basename(x))[0]

    Image.fromarray(result).save(
        f"Images/results/{name(first_image_path) + '_' + name(second_image_path)}.png"
    )

