import string

import font_clip
import pydiffvg
import streamlit as st
import torch
from tqdm import tqdm

from losses import *
from render import Render
from utils import *

losses_list = [AestheticLoss, NoveltyLoss]

letters = list(string.ascii_uppercase)

transform = transforms.ToPILImage()


def run():
    progress_text = "Operation in progress. Please wait."
    progress_bar = st.progress(0., text=progress_text)

    model, preprocess = font_clip.load("ViT-B/32", device=device)

    loss_functions = [CLIPLoss(f"The letter {letters[i]}", model=model, preprocess=preprocess) for i in range(26)]

    renders = [Render(letter=letters[i]) for i in range(26)]
    optims = [render.get_optim() for render in renders]

    for i in tqdm(range(iterations)):
        for optim in optims:
            if i == int(iterations * 0.5) or i == int(iterations * 0.75):
                for g in optim.param_groups:
                    g['lr'] /= 10

        # display_imgs = []
        for l in range(26):
            images = renders[l].render()

            # display_img = images[224].squeeze()
            # display_img = transform(display_img)
            # display_imgs.append(display_img)

            optims[l].zero_grad()

            loss = loss_functions[l](images[224])

            loss.backward()

            optims[l].step()

        # if i % 10 == 0:
            # image_container.image(display_imgs, width=100)

        progress_value = map_value(i, 0, iterations, 0.0, 1.0)
        progress_bar.progress(progress_value, text=progress_text)

    # pydiffvg.save_svg(f"{prompt}.svg", img_size, img_size, shapes, shape_groups)

    # st.image(filteredImages, width=150)


def debug():
    losses = []
    for loss, loss_enable in zip(losses_list, losses_list_enable):
        if loss_enable:
            losses.append(loss())

            print(losses[-1].get_classname())


# Setup things
st.set_page_config(page_title="Font Generation App", page_icon="💤", initial_sidebar_state="expanded", )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(device)

st.title("Font Generation Tool")

with st.sidebar:
    st.title("Settings")

    prompt_style = st.text_input(
        "Enter the prompt style",
        placeholder="Prompt",
    )

    iterations = st.number_input('Insert number of iterations', min_value=0, max_value=2000, step=100, value=1000)

    img_size = st.number_input('Insert canvas size', min_value=32, max_value=1024, step=1, value=224)


    # clip_model_checkboxs = []
    # with st.expander("Clip Models", expanded=True):
    #     for clip_model in clip_models_names:
    #         clip_model_checkboxs.append(st.checkbox(clip_model))

    losses_list_values = [0 for _ in losses_list]
    losses_list_enable = [False for _ in losses_list]
    target_image = None
    with st.expander("Losses", expanded=True):
        for l, loss in enumerate(losses_list):
            losses_list_enable[l] = st.checkbox(loss.get_classname())
            if losses_list_enable[l]:
                losses_list_values[l] = st.slider(f'{loss.get_classname()} value', 0.0, 1.0, 0.0)

    run_btn = st.button('Run', on_click=run)

with st.container():
    image_container = st.empty()

    save_btn = st.button('Save')
