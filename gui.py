import random
import string

import clip
import pydiffvg
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from losses import *
from render import SingleRender
from utils import *

losses_list = [AestheticLoss, CLIPSimilarityLoss, NoveltyLoss, ClassificationLoss, PerceptualLoss]

letters = list(string.ascii_uppercase)

transform = transforms.ToPILImage()


def run():
    # print(losses_list_enable)
    # print(losses_list_values)

    progress_text = "Operation in progress. Please wait."
    progress_bar = st.progress(0., text=progress_text)

    """
    losses = []
    for loss, loss_enable in zip(losses_list, losses_list_enable):
        if loss_enable:
            if loss == CLIPSimilarityLoss or loss == PerceptualLoss:
                l = loss(target_image)
            elif loss == ClassificationLoss:
                l = loss(letter=letter)
            else:
                l = loss()

            losses.append(l)

    if prompt is not None:
        losses.append(CLIPLoss(prompt))
    """

    model, preprocess = clip.load("ViT-B/32", device=device)

    loss_functions = [CLIPLoss(f"The letter {letters[i]}", model=model, preprocess=preprocess) for i in range(26)]

    renders = [SingleRender(canvas_size=100) for _ in range(26)]
    optims = [render.get_optim() for render in renders]

    for i in tqdm(range(iterations)):
        for optim in optims:
            if i == int(iterations * 0.5):
                for g in optim.param_groups:
                    g['lr'] /= 10
            if i == int(iterations * 0.75):
                for g in optim.param_groups:
                    g['lr'] /= 10

        display_imgs = []
        for l in range(26):
            img = renders[l].render()

            display_img = img.squeeze()
            display_img = transform(display_img)
            display_imgs.append(display_img)

            optims[l].zero_grad()

            loss = loss_functions[l](img)

            loss.backward()

            optims[l].step()

        if i % 10 == 0:
            image_container.image(display_imgs, width=100)

        progress_value = map_value(i, 0, iterations, 0.0, 1.0)
        progress_bar.progress(progress_value, text=progress_text)

        if i >= 30:
            return

    # pydiffvg.save_svg(f"{prompt}.svg", img_size, img_size, shapes, shape_groups)

    # st.image(filteredImages, width=150)


# Setup things
st.set_page_config(page_title="Font Generation App", page_icon="💤", initial_sidebar_state="expanded", )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(device)

st.title("Font Generation Tool")

with st.sidebar:
    st.title("Settings")

    prompt = st.text_input(
        "Enter the promp",
        placeholder="Prompt",
    )

    letter = st.selectbox(
        'Select the letter to evolve',
        list(string.ascii_uppercase))

    iterations = st.number_input('Insert number of iterations', min_value=0, max_value=2000, step=100, value=1000)

    img_size = st.number_input('Insert canvas size', min_value=32, max_value=1024, step=1, value=224)

    # cutouts_number = st.number_input('Insert number of cutouts', min_value=1, max_value=200, step=1, value=10)

    num_segments = st.number_input('Insert number of segments', min_value=1, max_value=100, step=1, value=10)

    clip_model_checkboxs = []
    with st.expander("Clip Models", expanded=True):
        for clip_model in clip_models_names:
            clip_model_checkboxs.append(st.checkbox(clip_model))

    losses_list_values = [0 for _ in losses_list]
    losses_list_enable = [False for _ in losses_list]
    target_image = None
    with st.expander("Losses", expanded=True):
        for l, loss in enumerate(losses_list):
            losses_list_enable[l] = st.checkbox(loss.get_classname())
            if losses_list_enable[l]:
                losses_list_values[l] = st.slider(f'{loss.get_classname()} value', 0.0, 1.0, 0.0)

                if loss == CLIPSimilarityLoss or loss == PerceptualLoss:
                    target_file = st.file_uploader("Choose the target image file", type=['png', 'jpg', 'jpeg'])
                    if target_file is not None:
                        target_image = Image.open(target_file)
                        st.image(target_image, use_column_width=True)

    # Verify that if CLIP similarity was chosen the file was uploaded
    is_runnable = (losses_list_enable[1] and target_image is None)

    run_btn = st.button('Run', on_click=run, disabled=is_runnable)

with st.container():
    image_container = st.empty()

    save_btn = st.button('Save')
