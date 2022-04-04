import argparse
from captcha.image import ImageCaptcha
from tqdm import tqdm
import os
import string
import random
imageCaptcha = ImageCaptcha()
import multiprocessing as mp

def get_random_string(length):
    # choose from all lowercase letter
    letters = "abcdefghjkmnpqrstuvwxyz" + "ABCDEFGHIJKLMNPQRSTUVWXYZ" + "23456789" + "     "
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def makeCaptchas(dst, amount):
    imageCaptcha2 = ImageCaptcha()
    for i in range(amount):
        somestring = get_random_string(8)
        imageCaptcha2.write(somestring, dst + "/" + somestring.lower().replace(" ", "") + ".png")
        if (i % 20):
            scale = random.uniform(1, 3)
            hscale = random.uniform(1, 1.2)
            imageCaptcha = ImageCaptcha(int(160 * scale), int(60 * scale * hscale), font_sizes=(int(42*scale), int(50*scale), int(56*scale)))

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

    processes = []
    n = args.n
    numthreads = 8
    for i in range(numthreads):
        amountToMake = args.n//numthreads
        n -= args.n//numthreads
        if n < 0:
            amountToMake += n
        processes.append(mp.Process(target=makeCaptchas, args=(args.dst, amountToMake)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("DONE")
    #for i in tqdm(range(args.n)):
    #    global imageCaptcha
    #    somestring = get_random_string(8)
    #    imageCaptcha.write(somestring, args.dst + "/" + somestring.lower().replace(" ", "") + ".png")
    #    if (i % 20):
    #        scale = random.uniform(1, 5)
    #        hscale = random.uniform(1, 1.2)
    #        imageCaptcha = ImageCaptcha(int(160 * scale), int(60 * scale * hscale), font_sizes=(int(42*scale), int(50*scale), int(56*scale)))



def captcha_samples():
    imageCaptcha.write("1a2b6", f"images/1a2b6.png")
    imageCaptcha.write("cdegh", f"images/cdegh.png")
    imageCaptcha.write("opqrstu", f"images/opqrstu.png")

    img = imageCaptcha.generate_image("12m5f")
    img.show()


if __name__ == "__main__":
    run()
