import random
import string

import numpy as np
import pydiffvg
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from losses import *
from utils import *

losses_list = [AestheticLoss, CLIPSimilarityLoss, NoveltyLoss, XingLoss]


transform = transforms.ToPILImage()


def run():
    # print(losses_list_enable)
    # print(losses_list_values)

    progress_text = "Operation in progress. Please wait."
    progress_bar = st.progress(0., text=progress_text)

    losses = []
    for loss, loss_enable in zip(losses_list, losses_list_enable):
        if loss_enable:
            if loss == CLIPSimilarityLoss:
                l = loss(target_image)
            else:
                l = loss()

            losses.append(l)

    if prompt is not None:
        losses.append(CLIPLoss(prompt))

    shapes = []
    shape_groups = []
    colors = []
    cell_size = int(img_size / 28)
    for r in range(28):
        cur_y = r * cell_size
        for c in range(28):
            cur_x = c * cell_size
            p0 = [cur_x, cur_y]
            p1 = [cur_x + cell_size, cur_y + cell_size]

            cell_color = torch.tensor([1.0, 1.0, 1.0, 1.0])
            colors.append(cell_color)

            path = pydiffvg.Rect(p_min=torch.tensor(p0), p_max=torch.tensor(p1))
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), stroke_color=None,
                                             fill_color=cell_color)
            shape_groups.append(path_group)

    color_vars = []
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_vars.append(group.fill_color)

    # Just some diffvg setup
    scene_args = pydiffvg.RenderFunction.serialize_scene(img_size, img_size, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply

    # optims = [torch.optim.Adam(points_vars, lr=1.0)]
    optims = [torch.optim.Adam(color_vars, lr=0.01)]

    for i in tqdm(range(iterations)):
        img = render(img_size, img_size, 2, 2, 0, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        for optim in optims:
            optim.zero_grad()

        loss_value = 0
        for loss in losses:
            loss_value += loss(img)
        loss_value.backward()

        for optim in optims:
            optim.step()

        # Clip values
        # print(color_vars)
        for group in shape_groups:
            temp_color = torch.mean(group.fill_color.data[:-1])
            # temp_color = torch.round(temp_color)
            temp_color = temp_color.repeat(3)
            group.fill_color.data = torch.cat((temp_color, torch.ones(1, requires_grad=True)))

        if i % 10 == 0:
            display_img = img.squeeze()
            display_img = transform(display_img)
            image_container.image(display_img)

        progress_value = map_value(i, 0, iterations, 0.0, 1.0)
        progress_bar.progress(progress_value, text=progress_text)

# Setup things
st.set_page_config(page_title="Font Generation App", page_icon="🧊", initial_sidebar_state="expanded", )

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

                if loss == CLIPSimilarityLoss:
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
