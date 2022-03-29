# coding: utf-8
"""
    captcha.image
    ~~~~~~~~~~~~~

    Generate Image CAPTCHAs, just the normal image CAPTCHAs you are using.
"""
from pillow_lut import rgb_color_enhance
import os
import random
from tkinter import W
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype

import math

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
try:
    from wheezy.captcha import image as wheezy_captcha
except ImportError:
    wheezy_captcha = None

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
DEFAULT_FONTS = [os.path.join(DATA_DIR, "DroidSansMono.ttf"), os.path.join(DATA_DIR, "Smokum-Regular.ttf"), os.path.join(DATA_DIR, "PlayfairDisplay-Black.ttf"), os.path.join(DATA_DIR, "COMIC.TTF")]#, "Smokum-Regular.ttf", "PlayfairDisplay-Black.ttf"

if wheezy_captcha:
    __all__ = ["ImageCaptcha", "WheezyCaptcha"]
else:
    __all__ = ["ImageCaptcha"]


table = []
for i in range(256):
    table.append(int(i * 1.97))

class WaveDeformer:
    def __init__(self, u, p):
        self.u = u
        self.p = p


    def transform(self, x, y):
        y = y + 10*math.sin(x/(40 - 35*self.u) + (self.p *self.u * 1000)) * self.p
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (*self.transform(x0, y0),
                *self.transform(x0, y1),
                *self.transform(x1, y1),
                *self.transform(x1, y0),
                )

    def getmesh(self, img):
        self.w, self.h = img.size
        gridspace = 10

        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))

        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]

        return [t for t in zip(target_grid, source_grid)]

class WaveDeformer2:
    def __init__(self, u, p):
        self.u = u
        self.p = p


    def transform(self, x, y):
        x = x + 5*math.sin(x/(20 - 18*self.u) + (self.p *self.u * 1000)) * self.p
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (*self.transform(x0, y0),
                *self.transform(x0, y1),
                *self.transform(x1, y1),
                *self.transform(x1, y0),
                )

    def getmesh(self, img):
        self.w, self.h = img.size
        gridspace = 10

        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))

        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]

        return [t for t in zip(target_grid, source_grid)]

class _Captcha(object):
    def generate(self, chars, format="png"):
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars, output, format="png", seed=None):
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        if seed is not None:
            random.seed(seed)

        im = self.generate_image(chars)
        return im.save(output, format=format)
    
    def write_clean(self, chars, output, format="png", seed=None):
        """Generate and write a clean image CAPTCHA data to the output.
        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        if seed is not None:
            random.seed(seed)

        im = self.generate_clean_image(chars)
        return im.save(output, format=format)


class WheezyCaptcha(_Captcha):
    """Create an image CAPTCHA with wheezy.captcha."""

    def __init__(self, width=200, height=75, fonts=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS

    def generate_image(self, chars):
        text_drawings = [
            wheezy_captcha.warp(),
            wheezy_captcha.rotate(),
            wheezy_captcha.offset(),
        ]
        fn = wheezy_captcha.captcha(
            drawings=[
                wheezy_captcha.background(),
                wheezy_captcha.text(fonts=self._fonts, drawings=text_drawings),
                wheezy_captcha.curve(),
                wheezy_captcha.noise(),
                wheezy_captcha.smooth(),
            ],
            width=self._width,
            height=self._height,
        )
        return fn(chars)


class ImageCaptcha(_Captcha):
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """

    def __init__(self, width=160, height=60, fonts=None, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or (42, 50, 56)
        self._truefonts = []
        self.scale = int(height / 60)

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple(
            [truetype(n, s) for n in self._fonts for s in self._font_sizes]
        )
        return self._truefonts

    
    def create_noise_curve(self, image, color):
        w, h = image.size
        x1 = random.randint(0, int(w / 5))
        x2 = random.randint(w - int(w / 5), w)
        y1 = random.randint(int(h / 5), h - int(h / 5))
        y2 = random.randint(y1, h - int(h / 5))
        points = [x1, y1, x2, y2]
        end = random.randint(160, 200)
        start = random.randint(0, 20)
        Draw(image).arc(points, start, end, fill=color, width=random.randint(1, self.scale))
        return image

    @staticmethod
    def create_noise_dots(image, color, width=3, number=30, scale=1):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            
            draw.line(((x1, y1), (x1 + random.randint(-2*scale, 2*scale), y1 + random.randint(-2*scale, 2*scale))), fill=color, width=random.randint(1, 4)*scale)
            number -= 1
        return image

    def create_captcha_image(self, chars, color, background):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = Image.new("RGB", (self._width, self._height), background)
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            im = Image.new("RGBA", (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)

            # warp
            dx = w * random.uniform(0.1, 0.3)
            dy = h * random.uniform(0.2, 0.3)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1,
                y1,
                -x1,
                h2 - y2,
                w2 + x2,
                h2 + y2,
                w2 - x2,
                -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), Image.QUAD, data)
            return im

        images = []
        for c in chars:
            if random.random() > 0.5:
                images.append(_draw_character(" "))
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        for im in images:
            w, h = im.size
            mask = im.convert("L").point(table)
            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            image = image.resize((self._width, self._height))

        return image


    def overlay_noise(self, im):
        if random.randint(0, 1):
            noise = Image.effect_noise((im.size[0], im.size[1]), random.uniform(10, 30))
        else:
            noise = Image.effect_noise((im.size[0]//random.randint(1, 10*self.scale), im.size[1]//random.randint(1, 10*self.scale)), random.uniform(10, 30))
            noise = noise.resize((im.size[0], im.size[1]))

        return Image.blend(im, ImageOps.colorize(noise, (255, 255, 255), (0, 0, 0)), random.uniform(0, 0.5)) 

    def do_random_filter(self, im):
        i = random.randint(0, 100)
        img = im
        if (i < 15):
            img = img.filter(ImageFilter.BoxBlur(2*self.scale))
        elif (i < 30):
            img = img.filter(ImageFilter.GaussianBlur(radius=1*self.scale))
        
        i = random.randint(0, 100)
        if (i < 20):
            img = img.filter(ImageFilter.UnsharpMask(radius=2*self.scale, percent=300, threshold=3))
        elif (i < 30):
            img = img.filter(ImageFilter.MinFilter(size=3))
        elif (i < 40):
            img = img.filter(ImageFilter.MaxFilter(size=3))
        elif (i < 55):
            img = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), scale=1))

        if (random.randint(0, 100)):
            img = ImageOps.invert(img)

        i = random.randint(0, 100)
        if (i < 33):
            img = img.filter(rgb_color_enhance(32, brightness=0, exposure=random.uniform(-1, 1),  warmth=random.uniform(-0.1, 0.1), saturation=random.uniform(-1, 2), vibrance=0, hue=random.uniform(0, 1), gamma=1.0, linear=False))
        elif (i < 67):
            img = ImageOps.grayscale(img)
            colourNew = (255 * random.randint(0, 1), 255 * random.randint(0, 1), 255 * random.randint(0, 1))
            altColourNew = (255 - colourNew[0],255 - colourNew[1],255 - colourNew[2])
            img = ImageOps.colorize(img, colourNew, altColourNew)


        i = random.randint(0, 100)
        if (i < 20):
            img = ImageOps.deform(img, WaveDeformer(random.uniform(0., 1.), random.uniform(0., 1.)))
        elif (i < 30):
            img = ImageOps.deform(img, WaveDeformer2(random.uniform(0., 1.), random.uniform(0., 1.)))
        elif (i < 40):
            img = ImageOps.deform(img, WaveDeformer(random.uniform(0., 1.), random.uniform(0., 1.)))
            img = ImageOps.deform(img, WaveDeformer2(random.uniform(0., 1.), random.uniform(0., 1.)))
            img = ImageOps.deform(img, WaveDeformer(random.uniform(0., 1.), random.uniform(0., 1.)))
        elif (i < 50):
            img = ImageOps.deform(img, WaveDeformer(random.uniform(0., 1.), random.uniform(0., 1.)))
            img = ImageOps.deform(img, WaveDeformer2(random.uniform(0., 1.), random.uniform(0., 1.)))
            img = ImageOps.deform(img, WaveDeformer(random.uniform(0., 1.), random.uniform(0., 1.)))
            img = ImageOps.deform(img, WaveDeformer2(random.uniform(0., 1.), random.uniform(0., 1.)))
            if (random.randint(0, 100)): # invert so that filler after distortion can be white
                img = ImageOps.invert(img)
            img = ImageOps.deform(img, WaveDeformer(random.uniform(0., 1.), random.uniform(0., 1.)))
            img = ImageOps.deform(img, WaveDeformer(random.uniform(0., 1.), random.uniform(0., 1.)))
            img = ImageOps.deform(img, WaveDeformer(random.uniform(0., 1.), random.uniform(0., 1.)))
            img = ImageOps.deform(img, WaveDeformer(random.uniform(0., 1.), random.uniform(0., 1.)))

        i = random.randint(0, 100)
        if (i < 70):
            img = self.overlay_noise(im)

        if (random.randint(0, 100)):
            img = ImageOps.invert(img)

        return img


    def generate_clean_image(self, chars):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = (255, 255, 255)
        color = (0, 0, 0, 255)
        im = self.create_captcha_image(chars, color, background)
        im = im.filter(ImageFilter.SMOOTH)
        return im

    def generate_image(self, chars):
        """Generate the clean image of the given characters.

        :param chars: text to be generated.
        """
        background = random_color(238, 255)
        color = random_color(10, 200, random.randint(220, 255))
        im = self.create_captcha_image(chars, color, background)
        self.create_noise_dots(im, color, number=random.randint(0, 100), scale=self.scale)
        for i in range(random.randint(0, 3)):
            self.create_noise_curve(im, color)
        im = im.filter(ImageFilter.SMOOTH)
        im = self.do_random_filter(im)
        self.create_noise_dots(im, color, number=random.randint(0, 20), scale=self.scale)
        return im

    def generate_char_image(self, chars, dst="chars", wid=38, height=75, dry_run=False):
        """
        Generates images of individual characters as if they were
        created by this same generator.

        dry_run: don't save images
        """
        background = random_color(238, 255)
        color = random_color(10, 200, random.randint(220, 255))
        image = Image.new("RGB", (wid, height), background)
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            im = Image.new("RGBA", (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)

            # warp
            dx = w * random.uniform(0.1, 0.3)
            dy = h * random.uniform(0.2, 0.3)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1,
                y1,
                -x1,
                h2 - y2,
                w2 + x2,
                h2 + y2,
                w2 - x2,
                -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), Image.QUAD, data)
            return im

        images = []
        for c in chars:
            images.append(_draw_character(c))

        # generate images of individual chars
        char_images = []
        clean_bg = image.copy()
        for i, im in enumerate(images):
            w, h = im.size
            mask = im.convert("L").point(table)
            image.paste(im, (0, int((height - h) / 2)), mask)
            color = random_color(10, 200, random.randint(220, 255))
            self.create_noise_dots(image, color)
            self.create_noise_curve(image, color)
            c = chars[i]
            if not dry_run:
                if not os.path.exists(f"{dst}/{c}"):
                    os.makedirs(f"{dst}/{c}")
                count = len(os.listdir(f"{dst}/{c}")) + 1
                image.save(f"{dst}/{c}/{c}-{count}.png")
            char_images.append(image)
            image = clean_bg.copy()

        return char_images


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)
