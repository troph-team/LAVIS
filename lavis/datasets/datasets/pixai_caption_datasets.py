import os.path as osp
from copy import deepcopy
import math
import numpy as np

from PIL import Image

from lavis.datasets.datasets.base_dataset import BaseDataset

from .nsfw_utils import is_nsfw_prompt


def pick_data(record):
    city = record["city"]
    country = record["country"]
    device_type = record["device_type"]
    platform = record["platform"]
    language = record["language"]
    event_time = record["event_time"]
    form_type = record["event_properties"]["formType"]
    preferred_prompt = record["event_properties"]["preferredPrompts"]
    disliked_prompt = record["event_properties"]["dislikedPrompts"]
    source = record["event_properties"]["source"]
    media_id = record["event_properties"]["mediaId"]
    media_url = record["event_properties"]["mediaUrl"]
    prompts = preferred_prompt + disliked_prompt
    is_nsfw = any(is_nsfw_prompt(prompt) for prompt in prompts)
    return {
        "city": city,
        "country": country,
        "device_type": device_type,
        "platform": platform,
        "language": language,
        "event_time": event_time,
        "form_type": form_type,
        "preferred_prompt": preferred_prompt,
        "disliked_prompt": disliked_prompt,
        "source": source,
        "media_id": media_id,
        "media_url": media_url,
        "is_nsfw": is_nsfw,
    }


class PixAICaptionDataset(BaseDataset):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 prompt, nsfw_prob=0):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.nsfw_prob = nsfw_prob
        annotations = []
        for anno in self.annotation:
            annotations.append(pick_data(anno))
        self.annotation = annotations
        self.prompt = prompt

    def load_datainfo(self):
        annotations = []
        for anno in self.annotation:
            data_info = pick_data(anno)
            is_nsfw = data_info['is_nsfw']
            rand = np.random.randn()
            if is_nsfw and rand > self.nsfw_prob:
                # skip for nsfw
                continue
            if not anno['preferred_prompt']:
                # skip for no preferred prompt
                continue

            if len(anno['disliked_prompt']) != 3:
                # skip for dislike prompts not equal to 3
                continue

            id_ = anno["media_id"]
            if not osp.exists(self.vis_root, f'{id_}.webp'):
                # skip for no data
                continue

        self.annotation = annotations

    def __getitem__(self, index):
        ann = self.annotation[index]

        id_ = ann["media_id"]
        # >>> just for test
        img = osp.join(self.vis_root, f'{id_}.webp')
        image = Image.open(img).convert("RGB")
        # <<< just for test
        # image = Image.open('docs/_static/Confusing-Pictures.jpg')
        image = self.vis_processor(image)

        pre_prompt = ann['preferred_prompt'][0].split(',')[0]
        answers = [pre_prompt]

        prompt = self.prompt

        answer_weight = {}
        for answer in answers:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(answers)
            else:
                answer_weight[answer] = 1 / len(answers)

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": prompt,
            "text_output": answers,
            "weights": weights,
        }
