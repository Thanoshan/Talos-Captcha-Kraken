"""General-purpose test script for image-to-image translation.

Adapted code from the pix2pix original authors to suit our purpose.

"""
import torch
from models.pix2pix.networks import define_G
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models.pix2pix.util import save_image, tensor2im

# Change gpu_id to -1 for CPU
def load_pix2pix_CAPTCHA(model_path="./checkpoints/GAN_Denoising/v3_net_G.pth", gpu_ids = [0], eval=False):
        model = define_G(
                input_nc=3,
                output_nc=3,
                ngf=64,
                netG="unet_256",
                norm="batch",
                use_dropout=True,
                init_type="normal",
                init_gain=0.02,
                gpu_ids=gpu_ids,
        )

        device = torch.device("cuda:{}".format(gpu_ids[0])) if gpu_ids else torch.device("cpu")

        state_dict = torch.load(model_path, map_location=str(device))

        model.load_state_dict(state_dict)
        
        if eval:
                model.eval()
        return model

def execute_pix2pix_denoise(model, img):
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

        with torch.no_grad():
                if torch.cuda.is_available():
                        result = (model(transform(img).cuda().unsqueeze(0)))
                else:
                        result = (model(transform(img).unsqueeze(0)))

                result_new = tensor2im(result)
                result_new = Image.fromarray(result_new).convert('RGB')
           #     save_image(result_new, "test.png")
        return result_new    

# test_path = "/Users/thanos/Documents/APS360_FinalProj_SRC/Talos-Captcha-Kraken/results/models/GAN_Denoising/images/ F7j62_real_A.png"
# test_image = Image.open(test_path).convert('RGB')
# test_image = np.asarray(test_image)
# net = load_pix2pix_CAPTCHA(gpu_ids=[])
# result = execute_pix2pix_denoise(net, test_image)



