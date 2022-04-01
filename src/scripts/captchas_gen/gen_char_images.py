import argparse
from captcha.image import ImageCaptcha
from tqdm import tqdm
import os
import string
import random
import time
imageCaptcha = ImageCaptcha()

def get_random_string(length):
    # choose from all lowercase letter
    letters = "abcdefghjkmnpqrstuvwxyz" + "ABCDEFGHIJKLMNPQRSTUVWXYZ" + "23456789" + "     "
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def run():
    """Run the generator with arguments:
    eg. python gen_char_images.py 100 --width=20 --height=75 --dst="trial"
    """
    parser = argparse.ArgumentParser(
        description="Generate single-character images from a captcha."
    )
    parser.add_argument(
        "n", type=int, help="number of images to generate for each character"
    )
    parser.add_argument(
        "-d", "--dry-run", action="store_true", help="run without saving images"
    )
    parser.add_argument(
        "-w", "--width", type=int, default=34, help="image width in pixels"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="chars",
        help="destination folder (see README for folder structure)",
    )
    parser.add_argument("--height", type=int, default=75, help="image height in pixels")
    parser.add_argument(
        "-c",
        "--chars",
        type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyz",
        help='characters to generate image for (eg. "adf")',
    )

    args = parser.parse_args()

    for i in tqdm(range(args.n)):
        os.makedirs(args.dst, exist_ok=True)
        global imageCaptcha
        somestring = get_random_string(random.randint(4, 9))
        seed = time.time()

        """Just captchas of dimension 320x144"""
        scale=2
        hscale=1.2
        imageCaptcha = ImageCaptcha(int(160 * scale), int(60 * scale * hscale), font_sizes=(int(42*scale), int(50*scale), int(56*scale)))
        imageCaptcha.write(somestring, args.dst + "/" + somestring + ".png")

        """Pairs of noisy and denoised images"""
        # os.makedirs(args.dst+"/clean", exist_ok=True)
        # os.makedirs(args.dst+"/noisy", exist_ok=True)
        # # imageCaptcha.write_clean(somestring, args.dst + "/clean/" + somestring + ".png", "png", seed)
        # imageCaptcha.write(somestring, args.dst + "/noisy/" + somestring + ".png", "png", seed)

        """Random resize of captchas"""
        # if (i % 20):
            # scale = random.uniform(1, 5)
            # hscale = random.uniform(1, 1.2)
            # imageCaptcha = ImageCaptcha(int(160 * scale), int(60 * scale * hscale), font_sizes=(int(42*scale), int(50*scale), int(56*scale)))

        """Generate character images"""
        # imageCaptcha.generate_char_image(
        #     args.chars,
        #     dst=args.dst,
        #     wid=args.width,
        #     height=args.height,
        #     dry_run=args.dry_run,
        # )


def captcha_samples():
    imageCaptcha.write("1a2b6", f"images/1a2b6.png")
    imageCaptcha.write("cdegh", f"images/cdegh.png")
    imageCaptcha.write("opqrstu", f"images/opqrstu.png")

    img = imageCaptcha.generate_image("12m5f")
    img.show()


if __name__ == "__main__":
    run()
