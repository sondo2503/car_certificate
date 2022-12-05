import json
import os
import os.path as osp
from tqdm import tqdm
from PIL import Image, ExifTags
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break
def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s
# folder_save_txt=r'label_pill_all'
path=r'M:\public_train\public_train\prescription'
for file in tqdm(os.listdir(osp.join(path, "label"))):
    txt = ''
    if file.endswith(".json"):


        with open(osp.join(path, "label", file),encoding='utf-8') as f:

                data = json.load(f)
                # print(data)
                # image_path = osp.join(path, "image", file.replace(".json", ".png"))
                # img = Image.open(image_path)

                name_pres, id = [None] * 2
                label_all=[]
                # print(data['txt'])
                for box in data:
                    if box['label']=='drugname':
                        # json_file=box['text'],box['mapping']
                        #
                        # print(json_file)
                        name_pres=box['text']
                        id=box['mapping']
                        json_file=str(name_pres),str(id)
                        label_all.append(json_file)
                for da in label_all:
                    print(da)
                    txt=str(da[0])+str("\t")+str(da[1])
                    with open(os.path.join('txt_file_name.txt'), 'a', encoding='utf-8') as f:
                        f.write(txt)
                        f.write('\n')
                        f.close()