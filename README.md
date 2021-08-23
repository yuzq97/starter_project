# An Attempt to Manipulate the Latent Space of GANs for Semantic Face Editing

For this starter project, I played with the latent space of StyleGAN (Karras *et al.*, CVPR 2019) and made an attempt at tackling the disentanglement of facial attributes, a task discussed in the original StyleGAN paper (https://arxiv.org/pdf/1812.04948.pdf). The purpose is to turn an unconditionally trained GAN model into a controllable one, which means that the model can edit a particular facial attribute without affecting another.

While the StyleGAN paper has already found that the intermediate latent space W is less entagled by Z, there exists another approach proposed in the CVPR paper "Interpreting the Latent Space of GANs for Semantic Face Editing" (https://arxiv.org/pdf/1907.10786.pdf) called InterFaceGan ((Shen *et al.*, CVPR 2020)). The authors of the paper first prove that the latent space Z of StyleGAN is separable by a hyperplane given any facial attributes, and then find a projected direction along which moving the latent code changes attribute A without affecting attribute B.

I implemented both approaches based on the work of InterFaceGan, and a demo can be found [[here](https://colab.research.google.com/github/yuzq97/starter_project/blob/main/demo.ipynb)].

## Instructions for Use of edit.py

Please download StyleGAN models from https://github.com/NVlabs/stylegan, and then put them under `pretrain/`. Both StyleGAN models trained on CelebA-HQ and FFHQ dataset are supported.

### Arguments:

-m: Model used for generating images, either "stylegan_ffhq" or "stylegan_celebahq". \
-o: Directory to save the output results. \
-b: Path to the semantic boundary. All boundaries are saved under `boundaries/`. \
-n: Number of images for editing. \
-s: Latent space used in StyleGAN, either "W" or "Z". ("Z" by default)

## Training Process
Due to the lack of a GPU, I used the pretrained StyleGAN models as well as pretrained unconditioned boundaries. Nonetheless, I was able to generate a few conditional boundaries using the utility function `project_boundary()`, which takes in a primal boundary and another one or two boundaries, and returns the modified primal boundary conditioned on the other boundaries.
