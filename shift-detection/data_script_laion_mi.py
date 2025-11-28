from tqdm.auto import tqdm
from datasets import load_dataset
import requests
from PIL import Image
import io
import pathlib
import numpy as np
import pandas as pd

timeout_seconds = 15

def get_image(image_url):

    try:

        response = requests.get(image_url, timeout=timeout_seconds)
        response.raise_for_status()
        
        image_data = io.BytesIO(response.content)

        img = np.array(
            Image.open(
                image_data
                )
        )

        if (3 == img.ndim) and (3 == img.shape[-1]):
            return img

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
    except IOError:
        print("Error opening or processing image data.")
    raise ValueError("")

if __name__ == "__main__":
	# Opening dataset
	dataset = load_dataset("antoniaaa/laion_mi")
	imgs = []

	for row in tqdm(dataset["members"]):
		try:
			imgs.append(
				{
				"image": get_image(row["url"]),
				"caption": row["caption"],
				"label": 1
				}
			)
			
		except ValueError:
			continue

	for row in tqdm(dataset["nonmembers"]):
		try:
			imgs.append(
				{
				"image": get_image(row["url"]),
				"caption": row["caption"],
				"label": 0
				}
			)
		except ValueError:
			continue

	image_path = pathlib.Path("data/laion_mi_image/images")
	image_path.mkdir(exist_ok=True, parents=True)

	image_paths = []
	for i, row in enumerate(tqdm(imgs)):
		cur_image_path = image_path.joinpath(f"image_{i}.jpg").as_posix()
		save_image_path = cur_image_path.split("/", 1)[-1]
		image_paths.append(save_image_path)

		Image.fromarray(row["image"]).convert("RGB").save(cur_image_path)

	df = pd.DataFrame.from_records(imgs)
	df["image_paths"] = image_paths
	df.drop(columns=["image"], inplace=True)
	df[["image_paths", "caption", "label"]].to_csv("data/laion_mi_image/laion_mi_image.csv", index=0)

	